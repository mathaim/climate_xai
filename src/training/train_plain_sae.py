#!/usr/bin/env python3
"""
Train K-sparse SAE on GraphCast layer activations.
Matches model.py architecture exactly.

Usage:
  python train_sae.py --layer 1
  python train_sae.py --layer 16
"""

import os
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class NPYLayerActivationStream(IterableDataset):
    def __init__(self, data_dir, layer_prefix, d_in=512, batch_size=8192, seed=0):
        super().__init__()
        files = sorted(glob.glob(os.path.join(data_dir, f"{layer_prefix}*.npy")))
        if not files:
            raise FileNotFoundError(
                f"No files found for prefix '{layer_prefix}' in {data_dir}"
            )

        self.d_in   = d_in
        self.batch  = batch_size
        self.seed   = seed
        self.file_meta = []

        for f in files:
            try:
                arr = np.load(f, mmap_mode="r")
            except Exception as e:
                print(f"[CORRUPT] {f}: {e}")
                continue
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr[:, 0, :]
            assert arr.ndim == 2 and arr.shape[1] == d_in, \
                f"Bad shape {arr.shape} in {f}"
            self.file_meta.append({"path": f, "n_nodes": arr.shape[0]})

        total = sum(m["n_nodes"] for m in self.file_meta)
        self.steps_per_epoch = math.ceil(total / batch_size)
        print(f"  {len(self.file_meta)} files | ~{total:,} nodes | "
              f"{self.steps_per_epoch} steps/epoch")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        nw  = worker.num_workers if worker else 1
        wid = worker.id          if worker else 0
        rng = np.random.default_rng(self.seed + 997 * wid)

        shard       = self.file_meta[wid::nw]
        max_batches = math.ceil(self.steps_per_epoch / nw)
        rng.shuffle(shard)
        yielded = 0

        for md in shard:
            if yielded >= max_batches:
                break
            X = np.load(md["path"], mmap_mode="r")
            if X.ndim == 3 and X.shape[1] == 1:
                X = X[:, 0, :]
            perm = rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], self.batch):
                if yielded >= max_batches:
                    return
                sel = perm[start:start + self.batch]
                if sel.size == 0:
                    break
                yield torch.from_numpy(X[sel])
                yielded += 1


# ── SAE (exact match to model.py) ─────────────────────────────────────────────
def topk(x, k: int):
    """Keep top-k per row, zero out the rest."""
    if k >= x.shape[1]:
        return x
    vals, idx = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x)
    mask.scatter_(1, idx, 1.0)
    return x * mask


class SAE(nn.Module):
    """Top-K Sparse Autoencoder with AuxK loss (OpenAI recipe)."""
    def __init__(self, d_in=512, latent=4096, k_active=32, k_aux=512,
                 unit_norm_decoder=True, dead_window=3_000_000):
        super().__init__()
        self.enc   = nn.Linear(d_in, latent, bias=False)
        self.dec   = nn.Linear(latent, d_in,  bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.k     = k_active
        self.k_aux = k_aux
        self.unit_norm_decoder = unit_norm_decoder
        self.dead_window       = dead_window
        self.eps               = 1e-8

        self.register_buffer("miss_counts", torch.zeros(latent, dtype=torch.long))
        self.register_buffer("dead_mask",   torch.zeros(latent, dtype=torch.bool))

        with torch.no_grad():
            nn.init.normal_(self.dec.weight, mean=0.0, std=1.0)
            W = self.dec.weight
            W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))
            self.enc.weight.copy_(W.t())
            self.b_pre.zero_()

    def forward(self, x):
        # normalise inputs
        x     = x - x.mean(dim=1, keepdim=True)
        x     = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
        # subtract shared pre-bias before encoding
        x_bar    = x - self.b_pre
        code_pre = torch.relu(self.enc(x_bar))
        code     = topk(code_pre, self.k)

        # decode
        if self.unit_norm_decoder:
            W     = self.dec.weight
            norms = W.norm(dim=0, keepdim=True).clamp_min(self.eps)
            recon = torch.addmm(self.b_pre, code, (W / norms).t())
        else:
            recon = self.dec(code) + self.b_pre

        # AuxK loss on dead latents
        if self.dead_mask.any():
            dead_code = code_pre * self.dead_mask.unsqueeze(0)
            aux_code  = topk(dead_code, min(self.k_aux, dead_code.shape[1]))
            aux_recon = torch.addmm(
                torch.zeros_like(self.b_pre), aux_code, self.dec.weight.t()
            )
        else:
            aux_recon = torch.zeros_like(x)

        return recon, code, aux_recon

    @torch.no_grad()
    def update_dead_mask(self, code, batch_size: int):
        active = (code > 0).any(dim=0).cpu()
        self.miss_counts[active]  = 0
        self.miss_counts[~active] += batch_size
        self.dead_mask = self.miss_counts >= self.dead_window


# ── Optimiser helpers (match model.py) ───────────────────────────────────────
@torch.no_grad()
def project_decoder_grads_orthogonal(model):
    W = model.dec.weight
    G = model.dec.weight.grad
    if G is None:
        return
    dots   = (G * W).sum(dim=0, keepdim=True)
    norms2 = (W * W).sum(dim=0, keepdim=True).clamp_min(1e-8)
    G.sub_((dots / norms2) * W)

@torch.no_grad()
def renorm_decoder_columns(model):
    W = model.dec.weight.data
    W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    # exact prefix the activation manager writes
    layer_prefix = f"layer{args.layer:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t"
    activations_dir = os.path.join(args.activations_dir, f"layer{args.layer:02d}")

    print(f"\n{'='*60}")
    print(f"Layer:          {args.layer}")
    print(f"Activations:    {activations_dir}")
    print(f"Prefix:         {layer_prefix}")
    print(f"Output:         {args.output_dir}")
    print(f"{'='*60}\n")

    dataset = NPYLayerActivationStream(
        data_dir=activations_dir,
        layer_prefix=layer_prefix,
        d_in=512,
        batch_size=8192,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)

    model = SAE(d_in=512, latent=4096, k_active=32, k_aux=512).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=2e-4, eps=6.25e-10)

    for epoch in range(args.epochs):
        total_loss = total_recon = total_aux = 0.0
        n = 0

        for xb in loader:
            xb = xb.to(device, dtype=torch.float32)
            recon, code, aux_recon = model(xb)

            # compute loss on normalised input (same normalisation as forward())
            x_norm = xb - xb.mean(dim=1, keepdim=True)
            x_norm = x_norm / x_norm.norm(dim=1, keepdim=True).clamp_min(1e-6)

            recon_loss = (x_norm - recon).pow(2).mean()
            aux_loss   = (x_norm - aux_recon).pow(2).mean() \
                         if model.dead_mask.any() \
                         else torch.tensor(0.0, device=device)
            loss = recon_loss + (1 / 32) * aux_loss

            opt.zero_grad()
            loss.backward()
            project_decoder_grads_orthogonal(model)
            opt.step()
            renorm_decoder_columns(model)
            model.update_dead_mask(code.detach(), xb.shape[0])

            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_aux   += aux_loss.item()
            n           += 1

        dead = model.dead_mask.sum().item()
        print(f"  Epoch {epoch+1:02d}/{args.epochs} | "
              f"loss={total_loss/n:.4f} | "
              f"recon={total_recon/n:.4f} | "
              f"aux={total_aux/n:.4f} | "
              f"dead={dead}/4096")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"sae_layer{args.layer:02d}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "d_in": 512, "latent": 4096, "k_active": 32,
            "k_aux": 512, "layer": args.layer,
        }
    }, ckpt_path)
    print(f"\n  Saved → {ckpt_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",          type=int, required=True,
                        help="GraphCast layer number (e.g. 1 or 16)")
    parser.add_argument("--activations_dir", type=str,
                        default="/scratch/euh7ys/graphcast_activations",
                        help="Base dir containing layer01/, layer16/ subfolders")
    parser.add_argument("--output_dir",     type=str,
                        default="/scratch/euh7ys/sae_checkpoints",
                        help="Where to save the trained SAE checkpoint")
    parser.add_argument("--epochs",         type=int, default=10)
    args = parser.parse_args()
    train(args)