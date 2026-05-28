#!/usr/bin/env python3
"""
Encode GraphCast layer 8 activations through the Matryoshka SAE.
Uses exact same forward pass as sae.py get_acts().
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

CKPT_PATH  = "/standard/AikyamLab/madelyn/GraphCast/MatryoshkaSAE/checkpoints_layer8_v4/final_model.pt"
ACT_DIR    = Path("/scratch/euh7ys/graphcast_activations_full")
OUTPUT_DIR = Path("/scratch/euh7ys/graphcast_matryoshka_latents_full")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
K_ACTIVE   = 32

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt["model_state_dict"]

W_enc        = state["W_enc"].to(DEVICE)         # [512, 4096]
b_enc        = state["b_enc"].to(DEVICE)         # [4096]
W_dec        = state["W_dec"].to(DEVICE)         # [4096, 512]
running_avg  = state["normalizer.running_avg"].to(DEVICE)  # [1]

d_in   = W_enc.shape[0]   # 512
latent = W_enc.shape[1]   # 4096

# Decoder column norms for scaling (matches get_acts einsum)
dec_norms = W_dec.norm(dim=1)  # [4096]

print(f"d_in={d_in}, latent={latent}, k={K_ACTIVE}")
print(f"running_avg={running_avg.item():.4f}")

def normalize(x):
    """RunningAvgNormalizer.normalize — matches sae.py exactly."""
    return x * (np.sqrt(d_in) / running_avg)

def unnormalize(acts):
    """RunningAvgNormalizer.unnormalize — matches sae.py exactly."""
    return acts * (running_avg / np.sqrt(d_in))

def encode(x):
    """Matches sae.py get_acts() exactly."""
    x = normalize(x)
    pre_acts = x @ W_enc + b_enc       # no ReLU
    # Per-sample top-K (get_acts uses target_l0 per sample not batch)
    topk_vals, topk_idx = torch.topk(pre_acts, K_ACTIVE, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_idx, topk_vals)
    # Scale by decoder column norms
    acts = acts * dec_norms
    acts = unnormalize(acts)
    return acts

# ── Encode all activation files ───────────────────────────────────────────────
act_files = sorted(ACT_DIR.glob("layer0008*.npy"))
print(f"\nFound {len(act_files)} activation files")

processed = skipped = 0

with torch.no_grad():
    for f in tqdm(act_files, desc="Encoding"):
        ts    = f.name.split('_t')[-1].replace('.npy', '')
        out_f = OUTPUT_DIR / f"matryoshka_encoded_t{ts}.npy"

        if out_f.exists():
            skipped += 1
            continue

        act = np.load(f)
        if act.ndim == 3:
            act = act.squeeze(1)

        if act.shape[1] != d_in:
            print(f"  WARNING: unexpected shape {act.shape} — skipping {f.name}")
            continue

        x       = torch.from_numpy(act).float().to(DEVICE)
        encoded = encode(x)
        np.save(out_f, encoded.cpu().numpy())
        processed += 1

print(f"\nDone — processed={processed}, skipped={skipped}")
