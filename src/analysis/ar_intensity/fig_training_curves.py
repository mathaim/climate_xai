"""Appendix training curves: reconstruction loss vs data seen (year-equivalents), per arch,
plus Matryoshka L0 saturation. Slices the final contiguous run from appended matryoshka logs."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
YR = 1461 * 40962.0   # node embeddings per year of 6-hourly data
RUNS = {"Layer 0": "/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer00",
        "Layer 8": "/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer08",
        "Layer 15": "/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer15"}
PLAIN = {"Layer 0": "/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer00",
         "Layer 8": "/scratch/euh7ys/climate_xai/checkpoints/plain_layer8",
         "Layer 15": "/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer15"}
COL = {"Layer 0": "#2980b9", "Layer 8": "#c0392b", "Layer 15": "#27ae60"}
def matry_final_run(path):
    rows = [json.loads(l) for l in open(f"{path}/training_log.jsonl")]
    starts = [i for i, r in enumerate(rows) if r["step"] == 500]
    return rows[starts[-1]:]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for lab, p in PLAIN.items():
    rows = [json.loads(l) for l in open(f"{p}/training_log.jsonl")]
    seen = {}
    for r in rows: seen[r["epoch"]] = r          # resume re-logs: keep last per epoch
    ep = sorted(seen); x = [e * seen[e]["steps"] * 8192 / YR for e in ep]
    ax[0].plot(x, [seen[e]["recon_loss"] for e in ep], "-o", ms=3, color=COL[lab], label=lab)
for lab, p in RUNS.items():
    rows = matry_final_run(p)
    x = [r["step"] * 4096 / YR for r in rows]
    ax[1].plot(x, [r["loss"] for r in rows], color=COL[lab], label=lab)
ax[0].set_title("(a) Standard Top-K SAE", fontsize=10, loc="left"); ax[0].set_ylabel("reconstruction loss")
ax[1].set_title("(b) Matryoshka Top-K SAE", fontsize=10, loc="left"); ax[1].set_yscale("log"); ax[1].set_ylabel("training loss (prefix sum)")
for a in ax: a.set_xlabel("year-equivalents of node embeddings"); a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/training_curves.png", dpi=170, bbox_inches="tight")
print("saved training_curves.png")
