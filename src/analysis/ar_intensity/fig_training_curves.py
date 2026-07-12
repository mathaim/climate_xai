"""Appendix training curves: all six SAEs on one common scale (approximate FVU)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
YR = 1461 * 40962.0
MATRY = {"Layer 0": "/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer00",
         "Layer 8": "/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer08",
         "Layer 15": "/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer15"}
PLAIN = {"Layer 0": "/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer00",
         "Layer 8": "/scratch/euh7ys/climate_xai/checkpoints/plain_layer8",
         "Layer 15": "/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer15"}
COL = {"Layer 0": "#2980b9", "Layer 8": "#c0392b", "Layer 15": "#27ae60"}
fig, ax = plt.subplots(figsize=(8, 5))
for lab, p in PLAIN.items():
    rows = [json.loads(l) for l in open(f"{p}/training_log.jsonl")]
    seen = {}
    for r in rows: seen[r["epoch"]] = r
    ep = sorted(seen)
    ax.plot([e * seen[e]["steps"] * 8192 / YR for e in ep],
            [seen[e]["recon_loss"] * 512 for e in ep], "-o", ms=4, color=COL[lab],
            label=f"Standard, {lab.lower()}")
for lab, p in MATRY.items():
    rows = [json.loads(l) for l in open(f"{p}/training_log.jsonl")]
    starts = [i for i, r in enumerate(rows) if r["step"] == 500]
    rows = rows[starts[-1]:]
    ax.plot([r["step"] * 4096 / YR for r in rows], [r["loss"] / 512.0 for r in rows],
            "--", color=COL[lab], label=f"Matryoshka, {lab.lower()}")
ax.set_xscale("log"); ax.set_ylim(0, 0.6)
ax.set_xlabel("year-equivalents of node embeddings (log scale)")
ax.set_ylabel("fraction of variance unexplained (approx.)")
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/training_curves.png", dpi=170, bbox_inches="tight")
print("saved training_curves.png")
