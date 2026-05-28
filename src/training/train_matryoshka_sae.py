#!/usr/bin/env python3
"""
Train Matryoshka SAE (BatchTopK) on GraphCast layer activations.

Usage:
  python -m src.training.train_matryoshka_sae --config configs/matryoshka_layer8_v4.json
  python -m src.training.train_matryoshka_sae --layer 1 --data_dir activations/layer01
  python -m src.training.train_matryoshka_sae --layer 8 --n_steps 300000 --batch_size 4096
"""

import torch
import numpy as np
import os
import time
import json
import argparse
from pathlib import Path
from glob import glob

from src.models.matryoshka_sae import MatryoshkaSAE


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_GROUP_SIZES = [256, 512, 1024, 2048, 4096]


def get_config():
    parser = argparse.ArgumentParser(description="Train MatryoshkaSAE on GraphCast activations")

    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_latents", type=int, default=4096)
    parser.add_argument("--group_sizes", type=int, nargs="+", default=DEFAULT_GROUP_SIZES,
                        help="Nested group boundaries (last must equal n_latents)")
    parser.add_argument("--target_l0", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=3e-2)

    # Training
    parser.add_argument("--n_steps", type=int, default=300_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    # Paths
    parser.add_argument("--layer", type=int, default=None,
                        help="GraphCast layer number (used to find activation files)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with activation .npy files (overrides --layer)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Where to save checkpoints")

    # Logging
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--metrics_every", type=int, default=5_000)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Config file (overrides defaults, CLI args override config file)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")

    args = parser.parse_args()

    # Load config file if provided, then override with CLI args
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        # Apply config values as defaults (CLI args take precedence)
        for k, v in config.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    # Resolve data_dir from layer if not explicitly set
    if args.data_dir is None:
        if args.layer is not None:
            args.data_dir = os.path.join("activations", f"layer{args.layer:02d}")
        else:
            parser.error("Must specify either --data_dir or --layer")

    return args


# ── Data loading ─────────────────────────────────────────────────────────────

class ActivationStreamingDataset:
    """Streams batches from .npy activation files on disk.

    Each file is (N_nodes, d_model). Files are loaded one at a time,
    rows shuffled, and served as batches.
    """

    def __init__(self, data_dir, d_model, batch_size, layer=None, seed=42):
        self.data_dir = Path(data_dir)
        self.d_model = d_model
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # Find activation files
        if layer is not None:
            prefix = f"layer{layer:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t"
            self.files = sorted(glob(str(self.data_dir / f"{prefix}*.npy")))
        else:
            self.files = sorted(glob(str(self.data_dir / "*.npy")))
            # Exclude stats files
            self.files = [f for f in self.files
                          if not any(x in f for x in ["feature_min", "feature_max", "feature_std"])]

        assert len(self.files) > 0, f"No .npy files found in {data_dir}"
        print(f"  Found {len(self.files)} data files")

        # Check normalization stats
        self.feat_min = None
        self.feat_max = None
        min_path = self.data_dir / "feature_min.npy"
        max_path = self.data_dir / "feature_max.npy"
        if min_path.exists() and max_path.exists():
            self.feat_min = np.load(min_path)
            self.feat_max = np.load(max_path)
            self.feat_range = self.feat_max - self.feat_min
            self.feat_range[self.feat_range < 1e-8] = 1.0
            print(f"  Loaded normalization stats (min/max)")

        # Shuffle file order
        self.rng.shuffle(self.files)
        self.file_idx = 0
        self.buffer = None
        self.buffer_idx = 0

    def _load_next_file(self):
        if self.file_idx >= len(self.files):
            self.rng.shuffle(self.files)
            self.file_idx = 0

        data = np.load(self.files[self.file_idx])
        if data.ndim == 3 and data.shape[1] == 1:
            data = data[:, 0, :]

        # Normalize if stats are available
        if self.feat_min is not None:
            data = 2.0 * (data - self.feat_min) / self.feat_range - 1.0

        # Shuffle rows
        perm = self.rng.permutation(data.shape[0])
        self.buffer = data[perm].astype(np.float32)
        self.buffer_idx = 0
        self.file_idx += 1

    def get_batch(self):
        if self.buffer is None or self.buffer_idx + self.batch_size > self.buffer.shape[0]:
            self._load_next_file()

        batch = self.buffer[self.buffer_idx:self.buffer_idx + self.batch_size]
        self.buffer_idx += self.batch_size
        return torch.from_numpy(batch).float()


# ── Training ─────────────────────────────────────────────────────────────────

def train(cfg):
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
    config_dict = {k: v for k, v in vars(cfg).items() if v is not None}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {output_dir / 'config.json'}")

    # Data
    print("Setting up data loader...")
    dataset = ActivationStreamingDataset(
        data_dir=cfg.data_dir,
        d_model=cfg.d_model,
        batch_size=cfg.batch_size,
        layer=cfg.layer,
        seed=cfg.seed,
    )

    # Model
    print("Initializing MatryoshkaSAE...")
    model = MatryoshkaSAE(
        d_model=cfg.d_model,
        n_latents=cfg.n_latents,
        group_sizes=cfg.group_sizes,
        target_l0=cfg.target_l0,
        n_steps=cfg.n_steps,
        lr=cfg.lr,
        permute_latents=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")
    print(f"  d_model={cfg.d_model}, n_latents={cfg.n_latents}")
    print(f"  group_sizes={cfg.group_sizes}")
    print(f"  target_l0={cfg.target_l0}, n_steps={cfg.n_steps}")

    # Resume from checkpoint
    start_step = 0
    if cfg.resume is not None:
        print(f"  Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        model.scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"] + 1
        model.step_count = start_step
        print(f"  Resumed at step {start_step}")

    # Training log
    log_file = output_dir / "training_log.jsonl"
    print(f"  Logging to {log_file}")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nStarting training from step {start_step}...")
    t0 = time.time()
    running_loss = 0.0
    running_l0 = 0.0

    for step in range(start_step, cfg.n_steps):
        batch = dataset.get_batch().to(device)

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
                "lr": model.scheduler.get_last_lr()[0],
                "steps_per_sec": steps_per_sec,
            }

            if return_metrics:
                for k, v in result.items():
                    if k.endswith("_fvu"):
                        log_entry[k] = float(v) if torch.is_tensor(v) else v

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(
                f"  Step {step + 1:>7d}/{cfg.n_steps} | "
                f"loss={avg_loss:.4f} | L0={avg_l0:.1f} | "
                f"{steps_per_sec:.1f} steps/s"
            )

            running_loss = 0.0
            running_l0 = 0.0
            t0 = time.time()

        # Save checkpoint
        if (step + 1) % cfg.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{step + 1:07d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "scheduler_state_dict": model.scheduler.state_dict(),
                "scaler_state_dict": model.scaler.state_dict(),
                "config": config_dict,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ───────────────────────────────────────────────────────
    final_path = output_dir / "final_model.pt"
    torch.save({
        "step": cfg.n_steps - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "scheduler_state_dict": model.scheduler.state_dict(),
        "scaler_state_dict": model.scaler.state_dict(),
        "config": config_dict,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
