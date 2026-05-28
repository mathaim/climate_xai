"""
Train a Matryoshka SAE on ERA5 data regridded to GraphCast's icosahedral mesh.

Data: ~57K .npy files, each (40962, 229) float32
      at /standard/AikyamLab/madelyn/GraphCast/GraphCastData/era5_inputs/
      Normalized to [-1, 1] per feature using precomputed min/max.

Model: MatryoshkaSAE from sae.py
       d_model=229, n_latents=4096, target_l0=75
"""

import torch
import numpy as np
import os
import sys
import time
import json
import argparse
from pathlib import Path
from glob import glob

# Add the SAE module directory to path
sys.path.insert(0, "/standard/AikyamLab/madelyn/GraphCast/MatryoshkaSAE")
from sae import MatryoshkaSAE


# ============================================================
# Config
# ============================================================

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=229)
    parser.add_argument("--n_latents", type=int, default=4096)
    parser.add_argument("--n_prefixes", type=int, default=8)
    parser.add_argument("--target_l0", type=float, default=75)
    parser.add_argument("--n_steps", type=int, default=300_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--sparsity_type", type=str, default="l1", choices=["l1", "log"])
    parser.add_argument("--min_prefix_length", type=int, default=16)

    parser.add_argument("--data_dir", type=str,
                        default="/standard/AikyamLab/madelyn/GraphCast/GraphCastData/era5_inputs")
    parser.add_argument("--output_dir", type=str,
                        default="/standard/AikyamLab/madelyn/GraphCast/MatryoshkaSAE/checkpoints")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--metrics_every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


# ============================================================
# Data loading
# ============================================================

class ERA5StreamingDataset:
    """
    Streams batches from random .npy files on disk.
    Each file is (40962, 229). We load one file at a time,
    shuffle its rows, normalize to [-1, 1], and serve batches.
    """
    def __init__(self, data_dir, batch_size, seed=42):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # Get all .npy files
        self.files = sorted(glob(str(self.data_dir / "era5_inputs_*.npy")))
        assert len(self.files) > 0, f"No .npy files found in {data_dir}"
        print(f"  Found {len(self.files)} data files")

        # Load normalization stats
        self.feat_min = np.load(self.data_dir / "feature_min.npy")   # (229,)
        self.feat_max = np.load(self.data_dir / "feature_max.npy")   # (229,)
        self.feat_range = self.feat_max - self.feat_min
        self.feat_range[self.feat_range < 1e-8] = 1.0  # Avoid division by zero
        print(f"  Loaded normalization stats (min/max)")
        print(f"  Feature min range: [{self.feat_min.min():.4f}, {self.feat_min.max():.4f}]")
        print(f"  Feature max range: [{self.feat_max.min():.4f}, {self.feat_max.max():.4f}]")

        # Shuffle file order
        self.rng.shuffle(self.files)
        self.file_idx = 0

        # Current buffer
        self.buffer = None
        self.buffer_idx = 0

    def _load_next_file(self):
        """Load the next file into the buffer, normalize, and shuffle rows."""
        if self.file_idx >= len(self.files):
            # Reshuffle and restart (new epoch)
            self.rng.shuffle(self.files)
            self.file_idx = 0

        data = np.load(self.files[self.file_idx])  # (40962, 229)

        # Normalize to [-1, 1]
        data = 2.0 * (data - self.feat_min) / self.feat_range - 1.0

        # Shuffle rows
        perm = self.rng.permutation(data.shape[0])
        self.buffer = data[perm].astype(np.float32)
        self.buffer_idx = 0
        self.file_idx += 1

    def get_batch(self):
        """Return a normalized batch of shape (batch_size, 229) as a torch tensor."""
        if self.buffer is None or self.buffer_idx + self.batch_size > self.buffer.shape[0]:
            self._load_next_file()

        batch = self.buffer[self.buffer_idx:self.buffer_idx + self.batch_size]
        self.buffer_idx += self.batch_size

        return torch.from_numpy(batch).float()


# ============================================================
# Training
# ============================================================

def train(cfg):
    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = vars(cfg)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {output_dir / 'config.json'}")

    # Data
    print("Setting up data loader...")
    dataset = ERA5StreamingDataset(cfg.data_dir, cfg.batch_size, seed=cfg.seed)

    # Model
    print("Initializing SAE...")
    model = MatryoshkaSAE(
        d_model=cfg.d_model,
        n_latents=cfg.n_latents,
        n_prefixes=cfg.n_prefixes,
        target_l0=cfg.target_l0,
        n_steps=cfg.n_steps,
        lr=cfg.lr,
        permute_latents=True,
        min_prefix_length=cfg.min_prefix_length,
        sparsity_type=cfg.sparsity_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")
    print(f"  d_model={cfg.d_model}, n_latents={cfg.n_latents}")
    print(f"  target_l0={cfg.target_l0}, n_prefixes={cfg.n_prefixes}")
    print(f"  n_steps={cfg.n_steps}, batch_size={cfg.batch_size}")
    print(f"  Normalization: min-max to [-1, 1]")

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume is not None:
        print(f"  Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        model.scaler.load_state_dict(ckpt["scaler_state_dict"])
        model.sparsity_controller.step = ckpt["sparsity_step"]
        model.sparsity_controller.sparsity_loss_scale = ckpt["sparsity_loss_scale"]
        start_step = ckpt["step"] + 1
        print(f"  Resumed at step {start_step}")

    # Training log
    log_file = output_dir / "training_log.jsonl"
    print(f"  Logging to {log_file}")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print(f"\nStarting training from step {start_step}...")
    t0 = time.time()
    running_loss = 0.0
    running_l0 = 0.0

    for step in range(start_step, cfg.n_steps):
        # Get batch
        batch = dataset.get_batch().to(device)

        # Training step (with detailed metrics periodically)
        return_metrics = (step % cfg.metrics_every == 0) and (step > 0)
        result = model.step(batch, return_metrics=return_metrics)

        running_loss += result["loss"].item()
        running_l0 += result["avg_l0"]

        # Log
        if (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = cfg.log_every / elapsed
            avg_loss = running_loss / cfg.log_every
            avg_l0 = running_l0 / cfg.log_every

            log_entry = {
                "step": step + 1,
                "loss": avg_loss,
                "avg_l0": float(avg_l0),
                "sparsity_scale": float(result["sparsity_scale"]),
                "lr": model.scheduler.get_last_lr()[0],
                "steps_per_sec": steps_per_sec,
            }

            # Add detailed metrics if available
            if return_metrics:
                for k, v in result.items():
                    if k.startswith("block_") or k.endswith("_fvu"):
                        log_entry[k] = float(v) if torch.is_tensor(v) else v

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(
                f"  Step {step+1:>7d}/{cfg.n_steps} | "
                f"loss={avg_loss:.4f} | L0={avg_l0:.1f} | "
                f"sparsity_scale={result['sparsity_scale']:.4f} | "
                f"{steps_per_sec:.1f} steps/s"
            )

            running_loss = 0.0
            running_l0 = 0.0
            t0 = time.time()

        # Save checkpoint
        if (step + 1) % cfg.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{step+1:07d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "scheduler_state_dict": model.scheduler.state_dict(),
                "scaler_state_dict": model.scaler.state_dict(),
                "sparsity_step": model.sparsity_controller.step,
                "sparsity_loss_scale": model.sparsity_controller.sparsity_loss_scale,
                "config": config_dict,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------
    final_path = output_dir / "final_model.pt"
    torch.save({
        "step": cfg.n_steps - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "scheduler_state_dict": model.scheduler.state_dict(),
        "scaler_state_dict": model.scaler.state_dict(),
        "sparsity_step": model.sparsity_controller.step,
        "sparsity_loss_scale": model.sparsity_controller.sparsity_loss_scale,
        "config": config_dict,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Summary stats
    print(f"\nFinal stats:")
    print(f"  Loss: {result['loss'].item():.4f}")
    print(f"  L0: {result['avg_l0']:.1f}")
    print(f"  Sparsity scale: {result['sparsity_scale']:.4f}")


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
