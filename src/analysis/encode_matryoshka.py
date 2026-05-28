#!/usr/bin/env python3
"""
Encode GraphCast layer activations through a trained Matryoshka SAE.
Uses exact same forward pass as MatryoshkaSAE.get_acts().

Usage:
  python -m src.analysis.encode_matryoshka \
    --checkpoint checkpoints/matryoshka_layer08_v4/final_model.pt \
    --activations_dir activations/layer08 \
    --layer 8 \
    --output_dir results/matryoshka_latents_layer08
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def load_encoder(checkpoint_path, device):
    """Load Matryoshka SAE weights for inference."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    W_enc = state["W_enc"].to(device)
    b_enc = state["b_enc"].to(device)
    W_dec = state["W_dec"].to(device)
    running_avg = state["normalizer.running_avg"].to(device)

    d_in = W_enc.shape[0]
    n_latents = W_enc.shape[1]
    dec_norms = W_dec.norm(dim=1)

    print(f"  d_in={d_in}, n_latents={n_latents}")
    print(f"  running_avg={running_avg.item():.4f}")

    return W_enc, b_enc, W_dec, dec_norms, running_avg, d_in, n_latents


def encode(x, W_enc, b_enc, dec_norms, running_avg, d_in, k_active):
    """Matches MatryoshkaSAE.get_acts() exactly."""
    # Normalize
    x_norm = x * (np.sqrt(d_in) / running_avg)
    # Encode
    pre_acts = x_norm @ W_enc + b_enc
    # Per-sample top-K
    topk_vals, topk_idx = torch.topk(pre_acts, k_active, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_idx, topk_vals)
    # Scale by decoder column norms
    acts = acts * dec_norms
    # Unnormalize
    acts = acts * (running_avg / np.sqrt(d_in))
    return acts


def main():
    parser = argparse.ArgumentParser(description="Encode activations through Matryoshka SAE")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained MatryoshkaSAE checkpoint")
    parser.add_argument("--activations_dir", type=str, required=True,
                        help="Directory with layer activation .npy files")
    parser.add_argument("--layer", type=int, required=True,
                        help="GraphCast layer number (for file prefix)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save encoded latent files")
    parser.add_argument("--k_active", type=int, default=32,
                        help="Top-K features to keep per sample")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # Load model
    W_enc, b_enc, W_dec, dec_norms, running_avg, d_in, n_latents = \
        load_encoder(args.checkpoint, device)

    # Find activation files
    act_dir = Path(args.activations_dir)
    prefix = f"layer{args.layer:04d}"
    act_files = sorted(act_dir.glob(f"{prefix}*.npy"))
    print(f"Found {len(act_files)} activation files")

    processed = skipped = 0

    with torch.no_grad():
        for f in tqdm(act_files, desc="Encoding"):
            ts = f.name.split('_t')[-1].replace('.npy', '')
            out_f = output_dir / f"matryoshka_encoded_t{ts}.npy"

            if out_f.exists():
                skipped += 1
                continue

            act = np.load(f)
            if act.ndim == 3:
                act = act.squeeze(1)

            if act.shape[1] != d_in:
                print(f"  WARNING: unexpected shape {act.shape} — skipping {f.name}")
                continue

            x = torch.from_numpy(act).float().to(device)
            encoded = encode(x, W_enc, b_enc, dec_norms, running_avg, d_in, args.k_active)
            np.save(out_f, encoded.cpu().numpy())
            processed += 1

    print(f"\nDone — processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
