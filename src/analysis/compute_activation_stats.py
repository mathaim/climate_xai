#!/usr/bin/env python3
"""
Compute min/max normalization stats for GraphCast layer 8 activations.

Scans all activation files, tracks per-feature min and max.
Saves feature_min.npy and feature_max.npy in the activation directory.

Usage:
  python compute_activation_stats.py \
    --data_dir /scratch/euh7ys/graphcast_activations_full
"""

import numpy as np
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with layer0008*.npy files")
    parser.add_argument("--n_files", type=int, default=None,
                        help="Max files to scan (None = all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(glob(str(data_dir / "layer0008*.npy")))

    if not files:
        # try sae_encoded format too
        files = sorted(glob(str(data_dir / "*.npy")))
        files = [f for f in files if "feature_min" not in f
                 and "feature_max" not in f and "feature_std" not in f]

    assert len(files) > 0, f"No activation files found in {data_dir}"

    if args.n_files:
        files = files[:args.n_files]

    print(f"Scanning {len(files)} files...")

    # Get dimension from first file
    sample = np.load(files[0])
    if len(sample.shape) == 3:
        sample = sample.squeeze()
    d_model = sample.shape[1]
    print(f"Activation dimension: {d_model}")

    feat_min = np.full(d_model, np.inf, dtype=np.float64)
    feat_max = np.full(d_model, -np.inf, dtype=np.float64)

    for f in tqdm(files, desc="Computing stats"):
        data = np.load(f)
        if len(data.shape) == 3:
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
