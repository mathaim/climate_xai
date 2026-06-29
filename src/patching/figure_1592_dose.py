"""Concept 1592 as a linear causal dial on GraphCast's predicted AR intensity (BC AR)."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
df = pd.read_csv(f"{D}/bc_ar_1592.csv"); piv = df.pivot(index="time", columns="cond", values="ivt_max")
alphas = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]; cols = [f"c1592_a{a:+.1f}" for a in alphas]
fig, ax = plt.subplots(figsize=(8.5, 6))
for t in piv.index:
    ax.plot(alphas, [piv.loc[t, c] for c in cols], "-", lw=1.0, alpha=0.45, color="#1b7837")
ym = [piv[c].mean() for c in cols]
ax.plot(alphas, ym, "-o", color="#0b3d0b", lw=2.6, ms=7, zorder=5, label="concept 1592 (mean over event)")
ax.plot([-1, 2], [piv["ctrl_a-1.0"].mean(), piv["ctrl_a+2.0"].mean()], "--s",
        color="#c0392b", lw=2, ms=7, label="control concept")
ax.axvline(0, color="#888", lw=0.8); ax.axhline(piv["c1592_a+0.0"].mean(), color="#888", lw=0.8, ls=":")
sl = (ym[-1] - ym[0]) / (alphas[-1] - alphas[0])
ax.annotate(f"~{sl:+.0f} kg m$^{{-1}}$s$^{{-1}}$ per unit\n(near-linear)", xy=(1.0, ym[4]),
            xytext=(-0.85, max(ym) * 0.97), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#0b3d0b"))
ax.set_xlabel("concept 1592 setting  (α:  −1 clamped off  →  0 baseline  →  +2 amplified)", fontsize=10)
ax.set_ylabel("predicted region max IVT  (kg m$^{-1}$ s$^{-1}$)", fontsize=10)
ax.set_title("A single SAE concept is a linear causal dial on GraphCast's predicted atmospheric river", fontsize=11.5)
ax.set_xticks(alphas); ax.legend(fontsize=9, loc="upper left")
out = "/scratch/euh7ys/climate_xai/plots/concept_1592_dose_response.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)
