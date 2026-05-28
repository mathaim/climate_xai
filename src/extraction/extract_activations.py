"""
Extract GraphCast layer activations for a range of timesteps.

Usage:
  python extract_activations.py --layer 0  --timestamps_file train_timestamps.txt --start_idx 0 --count 100
  python extract_activations.py --layer 7  --timestamps_file train_timestamps.txt --start_idx 0 --count 100
  python extract_activations.py --layer 15 --timestamps_file train_timestamps.txt --start_idx 0 --count 100
"""
import os
import argparse
import dataclasses
import functools
import time
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import gcsfs
import jax
import haiku as hk

from graphcast import (
    autoregressive, casting, checkpoint, data_utils,
    graphcast as graphcast_module, normalization, rollout,
)
from graphcast.deep_typed_graph_net import get_activation_manager


# ── Zarr cache ────────────────────────────────────────────────────────────────

_zarr_ds = None

def get_zarr_dataset():
    global _zarr_ds
    if _zarr_ds is None:
        print("  Opening Zarr store (one-time)...")
        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(
            "weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"
        )
        _zarr_ds = xr.open_zarr(store, consolidated=True)
        for old, new in [("latitude", "lat"), ("longitude", "lon")]:
            if old in _zarr_ds.coords:
                _zarr_ds = _zarr_ds.rename({old: new})
        if _zarr_ds.lat[0] > _zarr_ds.lat[-1]:
            _zarr_ds = _zarr_ds.reindex(lat=_zarr_ds.lat[::-1])
        print("  Zarr store ready.")
    return _zarr_ds


def load_era5_window(target_time):
    vars_keep = [
        "geopotential", "specific_humidity", "temperature",
        "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
        "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure", "total_precipitation_6hr",
        "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask",
    ]
    target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M")
    start_dt  = target_dt - timedelta(hours=24)
    ds = get_zarr_dataset()
    ds_window = ds.sel(time=slice(
        np.datetime64(start_dt.strftime("%Y-%m-%dT%H:%M")),
        np.datetime64(target_time),
    ))
    ds_window = ds_window[[v for v in vars_keep if v in ds_window.data_vars]].load()
    ds_window = ds_window.assign_coords(datetime=("time", ds_window.time.values))
    if len(ds_window.time) != 5:
        raise ValueError(f"Expected 5 timesteps, got {len(ds_window.time)}")
    return ds_window


# ── One-time GraphCast setup ──────────────────────────────────────────────────

def setup_graphcast():
    print("=" * 60)
    print("ONE-TIME SETUP")
    print("=" * 60)

    checkpoint_path = os.environ.get(
        "GRAPHCAST_CHECKPOINT",
        "graphcast_checkpoints/GraphCast - ERA5 1979-2017 - "
        "resolution 0.25 - pressure levels 37 - mesh 2to6 - "
        "precipitation input and output.npz"
    )
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast_module.CheckPoint)

    params       = ckpt.params
    state        = {}
    model_config = ckpt.model_config
    task_config  = ckpt.task_config

    from google.cloud import storage
    gcs    = storage.Client.create_anonymous_client()
    bucket = gcs.get_bucket("dm_graphcast")
    prefix = "graphcast/stats/"
    with bucket.blob(prefix + "diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev = xr.load_dataset(f).compute()
    with bucket.blob(prefix + "mean_by_level.nc").open("rb") as f:
        mean = xr.load_dataset(f).compute()
    with bucket.blob(prefix + "stddev_by_level.nc").open("rb") as f:
        stddev = xr.load_dataset(f).compute()

    def construct_wrapped_graphcast(model_config, task_config):
        predictor = graphcast_module.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean,
            stddev_by_level=stddev,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn): return functools.partial(fn, model_config=model_config, task_config=task_config)
    def with_params(fn):  return functools.partial(fn, params=params, state=state)
    def drop_state(fn):   return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
    am = get_activation_manager()

    print("Setup complete.\n")
    return run_forward_jitted, task_config, am


# ── Per-timestep extraction ───────────────────────────────────────────────────

def process_timestep(target_time, layer, run_forward_jitted, task_config, am, output_dir):
    time_str     = target_time.replace(":", "-")
    layer_prefix = f"layer{layer:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t"
    out_path     = os.path.join(output_dir, f"{layer_prefix}{time_str}.npy")

    if os.path.exists(out_path):
        return "skipped"

    ds = load_era5_window(target_time)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds, target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config),
    )
    am.__init__(
        enabled=True,
        save_dir=output_dir,
        save_steps=[layer],
        save_node_sets=["mesh_nodes"],
        mode="post_res",
    )
    am.set_time(time_str)
    rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs.expand_dims("batch", axis=0),
        targets_template=targets.expand_dims("batch", axis=0) * np.nan,
        forcings=forcings.expand_dims("batch", axis=0),
    )

    # Squeeze batch dim saved by activation manager: (40962, 1, 512) -> (40962, 512)
    if os.path.exists(out_path):
        arr = np.load(out_path)
        if arr.ndim == 3:
            np.save(out_path, arr.squeeze(1))

    return "done"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",           type=int, required=True)
    parser.add_argument("--timestamps_file", type=str, required=True)
    parser.add_argument("--start_idx",       type=int, default=0)
    parser.add_argument("--count",           type=int, default=100)
    parser.add_argument("--base_output_dir", type=str, default="activations")
    args = parser.parse_args()

    output_dir = os.path.join(args.base_output_dir, f"layer{args.layer:02d}")
    os.makedirs(output_dir, exist_ok=True)

    with open(args.timestamps_file) as f:
        all_timestamps = [l.strip() for l in f if l.strip()]
    timestamps = all_timestamps[args.start_idx : args.start_idx + args.count]

    print(f"Layer:      {args.layer}")
    print(f"Output dir: {output_dir}")
    print(f"Timestamps: {len(timestamps)}  ({timestamps[0]} → {timestamps[-1]})")
    print()

    setup_t0 = time.perf_counter()
    run_forward_jitted, task_config, am = setup_graphcast()
    setup_time = time.perf_counter() - setup_t0
    print(f"Setup: {setup_time:.1f}s\n")

    processed = skipped = errors = 0
    for i, ts in enumerate(timestamps):
        ts_t0 = time.perf_counter()
        try:
            result = process_timestep(ts, args.layer, run_forward_jitted, task_config, am, output_dir)
        except Exception as e:
            result = f"error: {e}"
        elapsed = time.perf_counter() - ts_t0

        if result == "done":      processed += 1
        elif result == "skipped": skipped   += 1
        else:                     errors    += 1

        if i % 10 == 0 or result.startswith("error"):
            print(f"  [{i+1}/{len(timestamps)}] {ts}: {result} ({elapsed:.1f}s)")

    total = time.perf_counter() - setup_t0
    print(f"\nDone in {total:.0f}s — processed={processed} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
    os._exit(0)
