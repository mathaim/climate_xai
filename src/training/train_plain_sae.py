#!/usr/bin/env python3
"""
Train Top-K Sparse Autoencoder on GraphCast layer activations.

Usage:
  python -m src.training.train_plain_sae --layer 8 \
    --data_dir /scratch/euh7ys/activations_layer08_train --epochs 10

  python -m src.training.train_plain_sae --layer 0 \
    --data_dir /standard/AikyamLab/madelyn/GraphCast/GraphCastData/layer00 --epochs 10
"""

import os
import glob
import math
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader

from src.models.plain_sae import (
    PlainSAE,
    project_decoder_grads_orthogonal,
    renorm_decoder_columns,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ──────────────────────────────────────────────────────────────────

class NPYActivationStream(IterableDataset):
    """Streams batches of activation vectors from .npy files on disk.

    Each file is (N_nodes, d_in). Files are memory-mapped and shuffled.
    """

    def __init__(self, data_dir, layer, d_in=512, batch_size=8192, seed=0):
        super().__init__()
        layer_prefix = f"layer{layer:04d}"
        files = sorted(glob.glob(os.path.join(data_dir, f"{layer_prefix}*.npy")))

        if not files:
            # Try loading all .npy files (excluding stats files)
            files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
            files = [f for f in files
                     if not any(x in f for x in ["feature_min", "feature_max", "feature_std"])]

        if not files:
            raise FileNotFoundError(
                f"No activation files found in {data_dir} "
                f"(tried prefix '{layer_prefix}' and all .npy)"
            )

        self.d_in = d_in
        self.batch_size = batch_size
        self.seed = seed
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
                f"Bad shape {arr.shape} in {f}, expected (*, {d_in})"
            self.file_meta.append({"path": f, "n_nodes": arr.shape[0]})

        total = sum(m["n_nodes"] for m in self.file_meta)
        self.steps_per_epoch = math.ceil(total / batch_size)
        print(f"  {len(self.file_meta)} files | ~{total:,} nodes | "
              f"{self.steps_per_epoch} steps/epoch")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        rng = np.random.default_rng(self.seed + 997 * wid)

        shard = self.file_meta[wid::nw]
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
            for start in range(0, X.shape[0], self.batch_size):
                if yielded >= max_batches:
                    return
                sel = perm[start:start + self.batch_size]
                if sel.size == 0:
                    break
                yield torch.from_numpy(X[sel])
                yielded += 1


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'=' * 60}")
    print(f"PlainSAE Training")
    print(f"  Layer:       {args.layer}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  d_in={args.d_in}, n_latents={args.n_latents}, k={args.k_active}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"{'=' * 60}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    dataset = NPYActivationStream(
        data_dir=args.data_dir,
        layer=args.layer,
        d_in=args.d_in,
        batch_size=args.batch_size,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)

    model = PlainSAE(
        d_in=args.d_in,
        n_latents=args.n_latents,
        k_active=args.k_active,
        k_aux=args.k_aux,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=6.25e-10)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}")

    # Training log
    log_file = output_dir / "training_log.jsonl"
    print(f"  Logging to {log_file}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        total_loss = total_recon = total_aux = 0.0
        n = 0

        for xb in loader:
            xb = xb.to(device, dtype=torch.float32)
            recon, code, aux_recon = model(xb)

            # Compute loss on normalized input (same normalization as forward)
            x_norm = xb - xb.mean(dim=1, keepdim=True)
            x_norm = x_norm / x_norm.norm(dim=1, keepdim=True).clamp_min(1e-6)

            recon_loss = (x_norm - recon).pow(2).mean()
            aux_loss = ((x_norm - aux_recon).pow(2).mean()
                        if model.dead_mask.any()
                        else torch.tensor(0.0, device=device))
            loss = recon_loss + (1 / 32) * aux_loss

            opt.zero_grad()
            loss.backward()
            project_decoder_grads_orthogonal(model)
            opt.step()
            renorm_decoder_columns(model)
            model.update_dead_mask(code.detach(), xb.shape[0])

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()
            n += 1

        elapsed = time.time() - t0
        dead = model.dead_mask.sum().item()
        avg_loss = total_loss / max(n, 1)
        avg_recon = total_recon / max(n, 1)
        avg_aux = total_aux / max(n, 1)

        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "aux_loss": avg_aux,
            "dead_latents": dead,
            "steps": n,
            "elapsed_sec": elapsed,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"  Epoch {epoch + 1:02d}/{args.epochs} | "
              f"loss={avg_loss:.4f} | "
              f"recon={avg_recon:.4f} | "
              f"aux={avg_aux:.4f} | "
              f"dead={dead}/{args.n_latents} | "
              f"{elapsed:.0f}s")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "config": config_dict,
            }, ckpt_path)
            print(f"    Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = output_dir / "final_model.pt"
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "config": config_dict,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlainSAE on GraphCast activations")
    parser.add_argument("--layer", type=int, required=True,
                        help="GraphCast layer number (for file prefix: layer{NNNN}_...)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with activation .npy files (direct path)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Where to save trained SAE checkpoints")
    parser.add_argument("--d_in", type=int, default=512,
                        help="Input activation dimension")
    parser.add_argument("--n_latents", type=int, default=4096,
                        help="SAE dictionary size")
    parser.add_argument("--k_active", type=int, default=32,
                        help="Top-K active features per sample")
    parser.add_argument("--k_aux", type=int, default=512,
                        help="Number of dead latents for AuxK loss")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)
