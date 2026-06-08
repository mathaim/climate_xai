#!/usr/bin/env python3
"""
Extract GraphCast layer activations — BATCHED version.

Loads GraphCast checkpoint, normalization stats, and JIT-compiles ONCE.
Then loops through all assigned timesteps, only re-loading the ERA5
window for each one.

Usage:
  python -m src.extraction.extract_activations_batch \
    --layer 8 --timestamps_file all_timestamps.txt \
    --start_idx 0 --count 100 --output_dir activations
"""

import os
import argparse
import numpy as np
import xarray as xr
import gcsfs
import jax
import haiku as hk
import functools
import dataclasses
import time
from datetime import datetime, timedelta
from graphcast import (
    autoregressive, casting, checkpoint, data_utils,
    graphcast as graphcast_module, normalization, rollout,
)
from graphcast.deep_typed_graph_net import get_activation_manager


# ============================================================================
# ERA5 loading (same as original, opens Zarr once per call)
# ============================================================================

# global zarr dataset — opened once, reused across timesteps
_zarr_ds = None

def get_zarr_dataset():
    """Open the Zarr store once and cache it."""
    global _zarr_ds
    if _zarr_ds is None:
        print("  Opening Zarr store (one-time)...")
        zarr_path = "gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"
        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(zarr_path[5:])
        _zarr_ds = xr.open_zarr(store, consolidated=True)
        if "latitude" in _zarr_ds.coords:
            _zarr_ds = _zarr_ds.rename(latitude="lat")
        if "longitude" in _zarr_ds.coords:
            _zarr_ds = _zarr_ds.rename(longitude="lon")
        if _zarr_ds.lat[0] > _zarr_ds.lat[-1]:
            _zarr_ds = _zarr_ds.reindex(lat=_zarr_ds.lat[::-1])
        print("  Zarr store ready.")
    return _zarr_ds


def load_era5_window(target_time):
    """Load 5 consecutive timesteps ending at target_time from cached Zarr."""
    vars_keep = [
        "geopotential", "specific_humidity", "temperature",
        "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
        "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure", "total_precipitation_6hr",
        "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"
    ]

    target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M")
    start_dt = target_dt - timedelta(hours=24)
    start_time = start_dt.strftime("%Y-%m-%dT%H:%M")
    end_time = target_time

    ds = get_zarr_dataset()
    ds_window = ds.sel(time=slice(np.datetime64(start_time), np.datetime64(end_time)))
    ds_window = ds_window[[v for v in vars_keep if v in ds_window.data_vars]].load()
    ds_window = ds_window.assign_coords(datetime=('time', ds_window.time.values))

    if len(ds_window.time) != 5:
        raise ValueError(f"Expected 5 timesteps, got {len(ds_window.time)} for {target_time}")

    return ds_window


# ============================================================================
# One-time setup: GraphCast model + JIT compilation
# ============================================================================

def setup_graphcast():
    """Load checkpoint, normalization stats, build and JIT-compile model.

    Returns: (run_forward_jitted, task_config, activation_manager)
    """
    print("=" * 60)
    print("ONE-TIME SETUP")
    print("=" * 60)

    # Load checkpoint
    print("Loading GraphCast checkpoint...")
    checkpoint_path = os.environ.get(
        "GRAPHCAST_CHECKPOINT",
        "graphcast_checkpoints/GraphCast - ERA5 1979-2017 - "
        "resolution 0.25 - pressure levels 37 - mesh 2to6 - "
        "precipitation input and output.npz"
    )
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast_module.CheckPoint)

    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    # Load normalization stats from GCS
    print("Loading normalization stats from GCS...")
    from google.cloud import storage
    gcs = storage.Client.create_anonymous_client()
    bucket = gcs.get_bucket("dm_graphcast")
    prefix = "graphcast/"

    with bucket.blob(prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev = xr.load_dataset(f).compute()
    with bucket.blob(prefix + "stats/mean_by_level.nc").open("rb") as f:
        mean = xr.load_dataset(f).compute()
    with bucket.blob(prefix + "stats/stddev_by_level.nc").open("rb") as f:
        stddev = xr.load_dataset(f).compute()

    # Build model
    print("Building and JIT-compiling model...")

    def construct_wrapped_graphcast(model_config, task_config):
        predictor = graphcast_module.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor, diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean, stddev_by_level=stddev)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    # Setup activation manager
    am = get_activation_manager()

    print("Setup complete.\n")
    return run_forward_jitted, task_config, am


# ============================================================================
# Process one timestep (uses pre-loaded model)
# ============================================================================

def process_timestep(target_time, layer, run_forward_jitted, task_config, am, output_dir):
    """Extract activations for one timestep using pre-loaded model."""

    time_str = target_time.replace(':', '-')

    # check if already done
    existing = [f for f in os.listdir(output_dir) if time_str in f]
    if existing:
        return "skipped"

    try:
        # load ERA5 window (the only per-timestep IO)
        ds = load_era5_window(target_time)

        # extract inputs/targets/forcings
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds,
            target_lead_times=slice("6h", "6h"),
            **dataclasses.asdict(task_config))

        inputs_batched = inputs.expand_dims('batch', axis=0)
        targets_batched = targets.expand_dims('batch', axis=0)
        forcings_batched = forcings.expand_dims('batch', axis=0)

        # configure activation saving for this timestep
        am.__init__(enabled=True, save_dir=output_dir, save_steps=[layer],
                    save_node_sets=["mesh_nodes"], mode="post_res", save_components=["mesh_gnn"])
        am.set_time(time_str)

        # run GraphCast (model already JIT-compiled, runs fast)
        predictions = rollout.chunked_prediction(
            run_forward_jitted, rng=jax.random.PRNGKey(0),
            inputs=inputs_batched,
            targets_template=targets_batched * np.nan,
            forcings=forcings_batched)

        return "done"

    except Exception as e:
        return f"error: {e}"


# ============================================================================
# Main: setup once, loop through timesteps
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch extract GraphCast activations (setup once, loop timesteps)")
    parser.add_argument("--layer", type=int, required=True,
                        help="GraphCast layer number to extract (e.g. 1, 8, 16)")
    parser.add_argument("--timestamps_file", type=str, required=True,
                        help="File with one timestamp per line (YYYY-MM-DDTHH:MM)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First line index in timestamps file")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of timesteps to process")
    parser.add_argument("--output_dir", type=str, default="activations",
                        help="Base output directory (layer subdir created automatically)")
    args = parser.parse_args()

    # Create layer-specific output subdirectory
    args.output_dir = os.path.join(args.output_dir, f"layer{args.layer:02d}")

    os.makedirs(args.output_dir, exist_ok=True)

    # read timestamp list
    with open(args.timestamps_file) as f:
        all_timestamps = [line.strip() for line in f if line.strip()]

    timestamps = all_timestamps[args.start_idx : args.start_idx + args.count]
    print(f"Processing {len(timestamps)} timesteps "
          f"(idx {args.start_idx} to {args.start_idx + len(timestamps) - 1})")
    print(f"  First: {timestamps[0]}")
    print(f"  Last:  {timestamps[-1]}")
    print()

    # ---- ONE-TIME SETUP ----
    setup_t0 = time.perf_counter()
    run_forward_jitted, task_config, am = setup_graphcast()
    setup_time = time.perf_counter() - setup_t0
    print(f"Setup took {setup_time:.1f}s\n")

    # ---- WARMUP: first timestep triggers JIT compilation ----
    print(f"Warmup (JIT compile on first timestep)...")
    warmup_t0 = time.perf_counter()
    result = process_timestep(timestamps[0], args.layer, run_forward_jitted,
                              task_config, am, args.output_dir)
    warmup_time = time.perf_counter() - warmup_t0
    print(f"  {timestamps[0]}: {result} ({warmup_time:.1f}s)\n")

    # ---- PROCESS REMAINING TIMESTEPS ----
    processed = 1 if result == "done" else 0
    skipped = 1 if result == "skipped" else 0
    errors = 0

    for i, ts in enumerate(timestamps[1:], start=1):
        ts_t0 = time.perf_counter()
        result = process_timestep(ts, args.layer, run_forward_jitted,
                                  task_config, am, args.output_dir)
        elapsed = time.perf_counter() - ts_t0

        if result == "done":
            processed += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1

        # progress every 10 timesteps
        if i % 10 == 0 or result.startswith("error"):
            print(f"  [{i+1}/{len(timestamps)}] {ts}: {result} ({elapsed:.1f}s)")

    total_time = time.perf_counter() - setup_t0
    print(f"\n{'='*60}")
    print(f"Done in {total_time:.0f}s")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    if processed > 0:
        avg = (total_time - setup_time - warmup_time) / max(processed + skipped - 1, 1)
        print(f"  Avg per timestep: {avg:.1f}s (after setup+warmup)")
        remaining = len(all_timestamps) - args.start_idx - len(timestamps)
        if remaining > 0:
            print(f"  Estimated remaining: {remaining * avg / 3600:.1f} GPU-hours")


if __name__ == "__main__":
    main()
    os._exit(0)  # force exit to prevent async cleanup errors
