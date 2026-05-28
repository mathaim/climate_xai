#!/usr/bin/env python3
"""
Compute min/max normalization stats for GraphCast layer activations.

Scans all activation files for a given layer, tracks per-feature min and max.
Saves feature_min.npy and feature_max.npy in the activation directory.

Usage:
  python -m src.analysis.compute_activation_stats \
    --data_dir activations/layer08 --layer 8

  python -m src.analysis.compute_activation_stats \
    --data_dir activations/layer01 --layer 1
"""

import numpy as np
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Compute per-feature min/max stats")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with activation .npy files")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer number (for file prefix filtering)")
    parser.add_argument("--n_files", type=int, default=None,
                        help="Max files to scan (None = all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Find activation files
    if args.layer is not None:
        prefix = f"layer{args.layer:04d}"
        files = sorted(glob(str(data_dir / f"{prefix}*.npy")))
    else:
        files = sorted(glob(str(data_dir / "*.npy")))
        files = [f for f in files
                 if not any(x in f for x in ["feature_min", "feature_max", "feature_std"])]

    assert len(files) > 0, f"No activation files found in {data_dir}"

    if args.n_files:
        files = files[:args.n_files]

    print(f"Scanning {len(files)} files...")

    # Get dimension from first file
    sample = np.load(files[0])
    if sample.ndim == 3:
        sample = sample.squeeze()
    d_model = sample.shape[1]
    print(f"Activation dimension: {d_model}")

    feat_min = np.full(d_model, np.inf, dtype=np.float64)
    feat_max = np.full(d_model, -np.inf, dtype=np.float64)

    for f in tqdm(files, desc="Computing stats"):
        data = np.load(f)
        if data.ndim == 3:
            data = data.squeeze()

        feat_min = np.minimum(feat_min, data.min(axis=0))
        feat_max = np.maximum(feat_max, data.max(axis=0))

    feat_min = feat_min.astype(np.float32)
    feat_max = feat_max.astype(np.float32)

    np.save(data_dir / "feature_min.npy", feat_min)
    np.save(data_dir / "feature_max.npy", feat_max)

    print(f"\nSaved to {data_dir}:")
    print(f"  feature_min.npy: shape={feat_min.shape}, range=[{feat_min.min():.4f}, {feat_min.max():.4f}]")
    print(f"  feature_max.npy: shape={feat_max.shape}, range=[{feat_max.min():.4f}, {feat_max.max():.4f}]")

    feat_range = feat_max - feat_min
    dead = (feat_range < 1e-8).sum()
    print(f"  Dead features (constant): {dead}/{d_model}")


if __name__ == "__main__":
    main()
