"""Concept 1592 as a linear causal dial: change in predicted AR intensity vs concept setting."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
df = pd.read_csv(f"{D}/bc_ar_1592.csv"); piv = df.pivot(index="time", columns="cond", values="ivt_max")
alphas = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]; cols = [f"c1592_a{a:+.1f}" for a in alphas]
base = piv["c1592_a+0.0"]
fig, ax = plt.subplots(figsize=(8.2, 6))
for t in piv.index:                                   # per-timestep, all through origin
    ax.plot(alphas, [piv.loc[t, c] - base[t] for c in cols], "-", lw=1.0, alpha=0.35, color="#1b7837")
ym = [(piv[c] - base).mean() for c in cols]
slope = (ym[-1] - ym[0]) / (alphas[-1] - alphas[0])
ax.plot(alphas, ym, "-o", color="#0b3d0b", lw=3, ms=8, zorder=5,
        label=f"concept 1592   (slope \u2248 +{slope:.0f} per unit)")
cd = [piv["ctrl_a-1.0"].mean() - base.mean(), piv["ctrl_a+2.0"].mean() - base.mean()]
ax.plot([-1, 2], cd, "--s", color="#c0392b", lw=2, ms=8, label="control concept")
ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
ax.set_xlabel("concept 1592 setting     (\u22121 clamped off    \u00b7    0 baseline    \u00b7    +2 amplified)", fontsize=10)
ax.set_ylabel("change in predicted region max IVT\n(vs baseline, kg m$^{-1}$ s$^{-1}$)", fontsize=10)
ax.set_title("Concept 1592 linearly controls GraphCast's predicted atmospheric-river intensity", fontsize=12)
ax.set_xticks(alphas); ax.legend(fontsize=10, loc="upper left")
out = "/scratch/euh7ys/climate_xai/plots/concept_1592_dose_response.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)
