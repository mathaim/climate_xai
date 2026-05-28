#!/usr/bin/env python3
"""
Encode GraphCast layer 8 activations through SAE — sparse output.

Supports both:
  --sae_type original  (top-k SAE from graphcast-interpretability)
  --sae_type matryoshka (Matryoshka SAE from MatryoshkaSAE/)

Saves only the top-k nonzero indices and values per node.
Output per file (npz):
  indices: (40962, k) int16
  values:  (40962, k) float32
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse


# ============================================================================
# Original SAE (top-k, from graphcast-interpretability)
# ============================================================================

class OriginalSAE(nn.Module):
    def __init__(self, d_in, latent, k_active):
        super().__init__()
        self.d_in = d_in
        self.latent = latent
        self.k_active = k_active
        self.enc = nn.Linear(d_in, latent, bias=False)
        self.dec = nn.Linear(latent, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))

    def encode_sparse(self, x):
        pre_acts = self.enc(x + self.b_pre)
        topk_values, topk_indices = torch.topk(pre_acts, self.k_active, dim=-1)
        return topk_indices, topk_values


# ============================================================================
# Matryoshka SAE (L1 sparsity, from MatryoshkaSAE/)
# ============================================================================

class MatryoshkaEncoder(nn.Module):
    """Minimal wrapper for inference only — loads Matryoshka SAE weights."""
    def __init__(self, d_model, n_latents, k_active=32):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k_active = k_active
        self.W_enc = nn.Parameter(torch.randn(d_model, n_latents))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(torch.randn(n_latents, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # normalizer
        self.running_avg = nn.Parameter(torch.ones(1), requires_grad=False)

    def encode_sparse(self, x):
        # normalize (same as RunningAvgNormalizer)
        x_norm = x * (np.sqrt(x.shape[-1]) / self.running_avg.detach())
        # encode
        pre_acts = x_norm @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)
        # scale by decoder norms (how Matryoshka SAE reports activations)
        dec_norms = self.W_dec.norm(dim=1)
        scaled_acts = acts * dec_norms
        # take top-k
        topk_values, topk_indices = torch.topk(scaled_acts, self.k_active, dim=-1)
        return topk_indices, topk_values


def load_sae(sae_path, sae_type, device):
    """Load SAE checkpoint and return model ready for inference."""
    checkpoint = torch.load(sae_path, map_location="cpu")

    if sae_type == "original":
        model_state = checkpoint["model_state"]
        d_in = model_state["enc.weight"].shape[1]
        latent = model_state["enc.weight"].shape[0]
        k_active = 32
        print(f"Original SAE: d_in={d_in}, latent={latent}, k_active={k_active}")
        model = OriginalSAE(d_in=d_in, latent=latent, k_active=k_active)
        model.load_state_dict(model_state, strict=False)

    elif sae_type == "matryoshka":
        # Matryoshka checkpoint has 'model_state_dict' key
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint

        d_model = model_state["W_enc"].shape[0]
        n_latents = model_state["W_enc"].shape[1]
        k_active = 32
        print(f"Matryoshka SAE: d_model={d_model}, n_latents={n_latents}, k_active={k_active}")

        model = MatryoshkaEncoder(d_model=d_model, n_latents=n_latents, k_active=k_active)
        # map state dict keys
        new_state = {}
        for k, v in model_state.items():
            if k == "normalizer.running_avg":
                new_state["running_avg"] = v
            elif k in ["W_enc", "b_enc", "W_dec", "b_dec"]:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)

    else:
        raise ValueError(f"Unknown sae_type: {sae_type}")

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--sae_type", type=str, default="matryoshka",
                        choices=["original", "matryoshka"])
    parser.add_argument("--layer", type=int, required=True,
                        help="GraphCast layer number (for file prefix)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"SAE type: {args.sae_type}")
    print(f"Layer: {args.layer}")
    model = load_sae(args.sae_path, args.sae_type, DEVICE)

    # find activation files
    act_dir = Path(args.activations_dir)
    prefix = f"layer{args.layer:04d}"
    act_files = sorted(act_dir.glob(f"{prefix}*.npy"))
    print(f"Found {len(act_files)} activation files")

    act_files = act_files[args.start_idx : args.start_idx + args.count]
    print(f"Processing {args.start_idx} to {args.start_idx + len(act_files) - 1}")

    processed = 0
    skipped = 0

    with torch.no_grad():
        for act_file in tqdm(act_files, desc="Encoding"):
            timestamp = act_file.name.split("_t")[-1].replace(".npy", "")
            out_file = output_dir / f"sae_sparse_t{timestamp}.npz"

            if out_file.exists():
                skipped += 1
                continue

            act_data = np.load(act_file)
            if len(act_data.shape) == 3:
                act_data = act_data.squeeze()

            n_nodes = act_data.shape[0]
            all_indices = []
            all_values = []

            for start in range(0, n_nodes, args.batch_size):
                end = min(start + args.batch_size, n_nodes)
                batch = torch.from_numpy(act_data[start:end]).float().to(DEVICE)
                indices, values = model.encode_sparse(batch)
                all_indices.append(indices.cpu().numpy())
                all_values.append(values.cpu().numpy())

            indices = np.concatenate(all_indices, axis=0).astype(np.int16)
            values = np.concatenate(all_values, axis=0).astype(np.float32)

            np.savez_compressed(out_file, indices=indices, values=values)
            processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
