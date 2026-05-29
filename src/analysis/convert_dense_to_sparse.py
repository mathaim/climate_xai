#!/usr/bin/env python3
"""
Convert dense SAE latent files (.npy) to sparse format (.npz).

Dense format: (40962, 4096) float32 — full latent vectors, mostly zeros
Sparse format: (40962, k) indices (int16) + (40962, k) values (float32)

Saves ~90% disk space. Uses GPU for fast top-k selection.

Usage:
  python -m src.analysis.convert_dense_to_sparse \
    --input_dir /scratch/euh7ys/latents_plain_dense_test \
    --output_dir /scratch/euh7ys/latents_plain_sparse_test \
    --k 32

  python -m src.analysis.convert_dense_to_sparse \
    --input_dir /scratch/euh7ys/latents_matryoshka_dense_test \
    --output_dir /scratch/euh7ys/latents_matryoshka_sparse_test \
    --k 32
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from glob import glob
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dense_to_sparse(dense_array, k):
    """Convert dense (N, D) array to top-k sparse indices and values using GPU."""
    x = torch.from_numpy(dense_array).to(DEVICE)
    topk_values, topk_indices = torch.topk(x.abs(), k, dim=1)
    # Gather actual (signed) values at the top-k positions
    topk_values = torch.gather(x, 1, topk_indices)
    indices = topk_indices.cpu().numpy().astype(np.int16)
    values = topk_values.cpu().numpy().astype(np.float32)
    return indices, values


def main():
    parser = argparse.ArgumentParser(description="Convert dense latents to sparse format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with dense .npy latent files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save sparse .npz files")
    parser.add_argument("--k", type=int, default=32,
                        help="Number of top-k values to keep per node")
    parser.add_argument("--delete_after", action="store_true",
                        help="Delete dense file after successful conversion")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob(str(input_dir / "*.npy")))
    print(f"Device: {DEVICE}")
    print(f"Found {len(files)} dense files in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Top-k: {args.k}")
    if args.delete_after:
        print("WARNING: --delete_after is set, dense files will be deleted after conversion")
    print()

    converted = skipped = errors = 0
    space_saved = 0

    for f in tqdm(files, desc="Converting"):
        f = Path(f)
        # Preserve filename but change extension
        out_name = f.stem + ".npz"
        out_path = output_dir / out_name

        if out_path.exists():
            skipped += 1
            continue

        try:
            dense = np.load(f)
            if dense.ndim == 3:
                dense = dense.squeeze()

            indices, values = dense_to_sparse(dense, args.k)
            np.savez_compressed(out_path, indices=indices, values=values)

            dense_size = f.stat().st_size
            sparse_size = out_path.stat().st_size
            space_saved += dense_size - sparse_size

            if args.delete_after:
                f.unlink()

            converted += 1

        except Exception as e:
            print(f"\n  ERROR on {f.name}: {e}")
            errors += 1

    print(f"\nDone!")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Space saved: {space_saved / 1e9:.1f} GB")
    if not args.delete_after and converted > 0:
        print(f"\n  Run again with --delete_after to remove dense files after verifying")


if __name__ == "__main__":
    main()
