"""GraphCast forward with optional L15 Plain-SAE clamp; returns the predicted next state.
Torch-free. Streams a 5-step ERA5 window from the Zarr for held-out events."""
import os, numpy as np, xarray as xr, jax, jax.numpy as jnp, haiku as hk, functools, dataclasses
from datetime import datetime, timedelta
import gcsfs
from graphcast import (autoregressive, casting, checkpoint, data_utils,
                       graphcast as gc, normalization, rollout)
from graphcast.deep_typed_graph_net import SAEInjector
from src.patching.sae_to_jax import load_sae, NPZ_L15
_CLAMP_SCALE = float(os.environ.get("CLAMP_SCALE", "1.0"))  # 1.0 = correct (delta * rownorm)
class CastInjector(SAEInjector):
    """SAEInjector reimplemented with threshold-based TopK (no scatter -> avoids the XLA
    block-limit blow-up) and output cast back to the input dtype (model runs in bfloat16)."""
    def __call__(self, x, alpha=None):
        if alpha is None:
            return x
        p = self.params
        xm = x - jnp.mean(x, axis=-1, keepdims=True)
        rownorm = jnp.linalg.norm(xm, ord=2, axis=-1, keepdims=True).clip(min=1e-6)
        xn = xm / rownorm
        code_pre = jax.nn.relu((xn - p.b_pre) @ p.enc_w)          # (..., 4096)
        vals, _ = jax.lax.top_k(code_pre, p.k_active)             # (..., k)
        code = jnp.where(code_pre >= vals[..., -1:], code_pre, 0.0)
        new_code = code * (1.0 + alpha)                           # alpha [4096] broadcasts over last axis
        if p.unit_norm_decoder:
            dec_eff = p.dec_w / jnp.linalg.norm(p.dec_w, axis=1, keepdims=True).clip(min=1e-8)
        else:
            dec_eff = p.dec_w
        delta = (new_code - code) @ dec_eff
        # map normalized-space delta back to the raw activation (~22x) -- the missing rescale
        return (x + delta * rownorm * _CLAMP_SCALE).astype(x.dtype)


class AddInjector(SAEInjector):
    """Additive injection: WRITE concepts into the activation (turn on silent concepts).
    alpha here = additive code vector beta (beta[c] = target activation level for concept c)."""
    def __call__(self, x, alpha=None):
        if alpha is None:
            return x
        p = self.params
        xm = x - jnp.mean(x, axis=-1, keepdims=True)
        rownorm = jnp.linalg.norm(xm, ord=2, axis=-1, keepdims=True).clip(min=1e-6)
        if p.unit_norm_decoder:
            dec_eff = p.dec_w / jnp.linalg.norm(p.dec_w, axis=1, keepdims=True).clip(min=1e-8)
        else:
            dec_eff = p.dec_w
        delta = alpha @ dec_eff
        return (x + delta * rownorm * _CLAMP_SCALE).astype(x.dtype)
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
def setup(npz=NPZ_L15):
    with open(os.environ["GRAPHCAST_CHECKPOINT"], "rb") as f:
        ck = checkpoint.load(f, gc.CheckPoint)
    from google.cloud import storage
    bkt = storage.Client.create_anonymous_client().get_bucket("dm_graphcast")
    def _st(n):
        with bkt.blob("graphcast/stats/" + n).open("rb") as f: return xr.load_dataset(f).compute()
    return dict(params=ck.params, state={}, mc=ck.model_config, tc=ck.task_config,
                diffs=_st("diffs_stddev_by_level.nc"), mean=_st("mean_by_level.nc"),
                stddev=_st("stddev_by_level.nc"), sae=load_sae(npz))
def make_forward(alpha, S, step=15, injector_cls=None):
    use = alpha is not None
    def construct(model_config, task_config):
        inj = (injector_cls or CastInjector)(S["sae"]) if use else None
        p = gc.GraphCast(model_config, task_config, mesh_sae_injector=inj,
                         mesh_sae_steps=([step] if use else None),
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
def build_inputs(target_time, S):
    ds = load_era5_window(target_time)
    inp, tar, frc = data_utils.extract_inputs_targets_forcings(
        ds, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(S["tc"]))
    return (inp.expand_dims('batch', axis=0),
            tar.expand_dims('batch', axis=0) * np.nan,
            frc.expand_dims('batch', axis=0))
def run_one(fwd, inp, tar, frc):
    return rollout.chunked_prediction(fwd, rng=jax.random.PRNGKey(0),
        inputs=inp, targets_template=tar, forcings=frc)
def predict(target_time, fwd, S):
    return run_one(fwd, *build_inputs(target_time, S))


# ---- Matryoshka field injection: add a precomputed raw-activation delta at the injection step ----
class FieldInjector(SAEInjector):
    """Add a fixed per-node delta field [nnode,1,512] (broadcasts over batch) at the injection step.
    The delta was precomputed in PyTorch to remove/insert a matryoshka concept's decoder contribution."""
    def __init__(self, field):
        try: super().__init__(field)
        except Exception: pass
        self.field = field
    def __call__(self, x, alpha=None):
        if alpha is None:
            return x
        return (x + self.field).astype(x.dtype)


class GateInjector(SAEInjector):
    """No-op injector that prints ||x_forward - x8_ref|| at the injection step, to confirm the forward's
    step activation equals the extracted x8 the delta was built from. Returns x unchanged."""
    def __init__(self, x8_ref):
        try: super().__init__(x8_ref)
        except Exception: pass
        self.x8_ref = np.asarray(x8_ref, np.float32)
    def __call__(self, x, alpha=None):
        if alpha is None:
            return x
        ref = self.x8_ref
        def _cb(xv):
            xv = np.asarray(xv, np.float32).reshape(-1, ref.shape[-1]); r = ref.reshape(-1, ref.shape[-1])
            n = min(xv.shape[0], r.shape[0])
            d = float(np.linalg.norm((xv[:n] - r[:n]).ravel())); rr = float(np.linalg.norm(r[:n].ravel()))
            print(f"[GATE] xshape={tuple(np.asarray(xv).shape)}  ||x_fwd - x8||={d:.3f}  rel={d/max(rr,1e-9):.4f}", flush=True)
        jax.debug.callback(_cb, x)
        return x


def make_field_forward(field, S, step=8, gate_x8=None):
    def construct(model_config, task_config):
        inj = GateInjector(gate_x8) if gate_x8 is not None else FieldInjector(field)
        p = gc.GraphCast(model_config, task_config, mesh_sae_injector=inj,
                         mesh_sae_steps=[step], mesh_sae_node_sets=["mesh_nodes"], mesh_sae_alpha=jnp.zeros(1))
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


class FieldCaptureInjector(SAEInjector):
    """Inject a fixed field at the FIRST configured mesh_sae_step; capture (pass-through) the
    activation at the SECOND step into `holder`. Use with mesh_sae_steps=[inject_step, capture_step];
    call ordering is resolved at trace time via a Python counter (inject@8 then capture@15)."""
    def __init__(self, field, holder):
        try: super().__init__(field)
        except Exception: pass
        self.field = field; self.holder = holder; self._calls = 0
    def __call__(self, x, alpha=None):
        if alpha is None:
            return x
        self._calls += 1
        if self._calls == 1:
            return (x + self.field).astype(x.dtype)
        jax.debug.callback(lambda xv: self.holder.append(np.asarray(xv, np.float32)), x)
        return x


def make_field_capture_forward(field, S, holder, inject_step=8, capture_step=15):
    def construct(model_config, task_config):
        inj = FieldCaptureInjector(field, holder)
        p = gc.GraphCast(model_config, task_config, mesh_sae_injector=inj,
                         mesh_sae_steps=[inject_step, capture_step], mesh_sae_node_sets=["mesh_nodes"], mesh_sae_alpha=jnp.zeros(1))
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


class AlphaCaptureInjector(CastInjector):
    """CastInjector (in-JAX encode + alpha edit) at the FIRST configured step; capture the
    activation at the SECOND. Use mesh_sae_steps=[edit_step, capture_step]."""
    def __init__(self, params, holder):
        super().__init__(params); self.holder = holder; self._calls = 0
    def __call__(self, x, alpha=None):
        if alpha is None: return x
        self._calls += 1
        if self._calls == 1: return super().__call__(x, alpha)
        jax.debug.callback(lambda xv: self.holder.append(np.asarray(xv, np.float32)), x)
        return x


class AddCaptureInjector(AddInjector):
    """AddInjector (write concepts in) at the FIRST step; capture at the SECOND."""
    def __init__(self, params, holder):
        super().__init__(params); self.holder = holder; self._calls = 0
    def __call__(self, x, alpha=None):
        if alpha is None: return x
        self._calls += 1
        if self._calls == 1: return super().__call__(x, alpha)
        jax.debug.callback(lambda xv: self.holder.append(np.asarray(xv, np.float32)), x)
        return x


def make_capture_forward(alpha, S, holder, injector_cls, edit_step=8, capture_step=15):
    def construct(model_config, task_config):
        inj = injector_cls(S["sae"], holder)
        p = gc.GraphCast(model_config, task_config, mesh_sae_injector=inj,
                         mesh_sae_steps=[edit_step, capture_step], mesh_sae_node_sets=["mesh_nodes"],
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


class CaptureOnlyInjector(SAEInjector):
    """Capture the activation at the configured step; modify nothing."""
    def __init__(self, holder):
        try: super().__init__(holder)
        except Exception: pass
        self.holder = holder
    def __call__(self, x, alpha=None):
        if alpha is None: return x
        jax.debug.callback(lambda xv: self.holder.append(np.asarray(xv, np.float32)), x)
        return x


def make_captureonly_forward(S, holder, step=8):
    def construct(model_config, task_config):
        p = gc.GraphCast(model_config, task_config, mesh_sae_injector=CaptureOnlyInjector(holder),
                         mesh_sae_steps=[step], mesh_sae_node_sets=["mesh_nodes"], mesh_sae_alpha=jnp.zeros(1))
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
