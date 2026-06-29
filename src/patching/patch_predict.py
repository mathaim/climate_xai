"""GraphCast forward with optional L15 Plain-SAE clamp; returns the predicted next state.
Torch-free. Streams a 5-step ERA5 window from the Zarr for held-out events."""
import os, numpy as np, xarray as xr, jax, jax.numpy as jnp, haiku as hk, functools, dataclasses
from datetime import datetime, timedelta
import gcsfs
from graphcast import (autoregressive, casting, checkpoint, data_utils,
                       graphcast as gc, normalization, rollout)
from graphcast.deep_typed_graph_net import SAEInjector
from src.patching.sae_to_jax import load_l15_sae
class CastInjector(SAEInjector):
    """SAEInjector reimplemented with threshold-based TopK (no scatter -> avoids the XLA
    block-limit blow-up) and output cast back to the input dtype (model runs in bfloat16)."""
    def __call__(self, x, alpha=None):
        if alpha is None:
            return x
        p = self.params
        xm = x - jnp.mean(x, axis=1, keepdims=True)
        xn = xm / jnp.linalg.norm(xm, ord=2, axis=1, keepdims=True).clip(min=1e-6)
        code_pre = jax.nn.relu((xn - p.b_pre) @ p.enc_w)
        vals, _ = jax.lax.top_k(code_pre, p.k_active)
        code = jnp.where(code_pre >= vals[:, -1:], code_pre, 0.0)
        new_code = code * (1.0 + alpha[None, :])
        if p.unit_norm_decoder:
            dec_eff = p.dec_w / jnp.linalg.norm(p.dec_w, axis=1, keepdims=True).clip(min=1e-8)
        else:
            dec_eff = p.dec_w
        return (x + (new_code - code) @ dec_eff).astype(x.dtype)
ZARR = "gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"
VARS = ["geopotential","specific_humidity","temperature","u_component_of_wind","v_component_of_wind",
        "vertical_velocity","2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
        "mean_sea_level_pressure","total_precipitation_6hr","toa_incident_solar_radiation",
        "geopotential_at_surface","land_sea_mask"]
_zarr = None
def _get_zarr():
    global _zarr
    if _zarr is None:
        fs = gcsfs.GCSFileSystem(token="anon"); ds = xr.open_zarr(fs.get_mapper(ZARR[5:]), consolidated=True)
        if "latitude" in ds.coords: ds = ds.rename(latitude="lat")
        if "longitude" in ds.coords: ds = ds.rename(longitude="lon")
        if ds.lat[0] > ds.lat[-1]: ds = ds.reindex(lat=ds.lat[::-1])
        _zarr = ds
    return _zarr
def load_era5_window(target_time):
    tdt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M")
    s = (tdt - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M")
    ds = _get_zarr().sel(time=slice(np.datetime64(s), np.datetime64(target_time)))
    ds = ds[[v for v in VARS if v in ds.data_vars]].load()
    ds = ds.assign_coords(datetime=('time', ds.time.values))
    assert len(ds.time) == 5, f"expected 5 steps, got {len(ds.time)} for {target_time}"
    return ds
def setup():
    with open(os.environ["GRAPHCAST_CHECKPOINT"], "rb") as f:
        ck = checkpoint.load(f, gc.CheckPoint)
    from google.cloud import storage
    bkt = storage.Client.create_anonymous_client().get_bucket("dm_graphcast")
    def _st(n):
        with bkt.blob("graphcast/stats/" + n).open("rb") as f: return xr.load_dataset(f).compute()
    return dict(params=ck.params, state={}, mc=ck.model_config, tc=ck.task_config,
                diffs=_st("diffs_stddev_by_level.nc"), mean=_st("mean_by_level.nc"),
                stddev=_st("stddev_by_level.nc"), sae=load_l15_sae())
def make_forward(alpha, S):
    use = alpha is not None
    def construct(model_config, task_config):
        inj = CastInjector(S["sae"]) if use else None
        p = gc.GraphCast(model_config, task_config, mesh_sae_injector=inj,
                         mesh_sae_steps=([15] if use else None),
                         mesh_sae_node_sets=(["mesh_nodes"] if use else None),
                         mesh_sae_alpha=alpha)
        p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=S["diffs"],
                                             mean_by_level=S["mean"], stddev_by_level=S["stddev"])
        return autoregressive.Predictor(p, gradient_checkpointing=True)
    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        return construct(model_config, task_config)(inputs, targets_template=targets_template, forcings=forcings)
    with_configs = lambda fn: functools.partial(fn, model_config=S["mc"], task_config=S["tc"])
    with_params = lambda fn: functools.partial(fn, params=S["params"], state=S["state"])
    drop_state = lambda fn: (lambda **kw: fn(**kw)[0])
    return drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
def predict(target_time, fwd, S):
    ds = load_era5_window(target_time)
    inp, tar, frc = data_utils.extract_inputs_targets_forcings(
        ds, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(S["tc"]))
    return rollout.chunked_prediction(fwd, rng=jax.random.PRNGKey(0),
        inputs=inp.expand_dims('batch', axis=0),
        targets_template=tar.expand_dims('batch', axis=0) * np.nan,
        forcings=frc.expand_dims('batch', axis=0))
