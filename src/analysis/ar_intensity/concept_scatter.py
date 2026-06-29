"""Small-multiples density (parent/child concept x region): 2D density of IVT vs activation.
Shared log colour scale. Matryoshka L8 concepts 99 (parent), 3153 & 3483 (children)."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"
REGIONS = [("W_N_America", "W.N.Am"), ("W_Europe", "W.Eu"), ("W_S_America", "W.S.Am"), ("E_Australia", "E.Aus")]
CONCEPTS = [(99, "Parent (99)"), (3153, "Child (3153)"), (3483, "Child (3483)")]
data = {}
for c, _ in CONCEPTS:
    for rkey, _ in REGIONS:
        t = np.load(f"{TRACK}/track_matry_{rkey}.npz")
        A = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float)
        ok = np.isfinite(ivt) & (A > 0); data[(c, rkey)] = (ivt[ok], A[ok])
XMAX = max(d[0].max() for d in data.values()) * 1.03
rowymax = {c: np.concatenate([data[(c, r)][1] for r, _ in REGIONS]).max() * 1.05 for c, _ in CONCEPTS}
fig, axes = plt.subplots(3, 4, figsize=(15, 9.8), sharex=True, sharey="row", constrained_layout=True)
hbs = []
for i, (c, clab) in enumerate(CONCEPTS):
    ymax = rowymax[c]
    for j, (rkey, rlab) in enumerate(REGIONS):
        ax = axes[i, j]; x, y = data[(c, rkey)]
        if len(x):
            hb = ax.hexbin(x, y, gridsize=30, cmap="viridis", mincnt=1, extent=(0, XMAX, 0, ymax)); hbs.append(hb)
        ax.text(0.96, 0.94, f"n={len(x):,}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2))
        ax.set_xlim(0, XMAX); ax.set_ylim(0, ymax)
        if i == 0: ax.set_title(rlab, fontsize=12)
        if j == 0: ax.set_ylabel(f"{clab}\n\nactivation", fontsize=10)
        if i == 2: ax.set_xlabel("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=10)
gmax = max((hb.get_array().max() for hb in hbs), default=1)
nrm = LogNorm(vmin=1, vmax=gmax)
for hb in hbs: hb.set_norm(nrm)
cb = fig.colorbar(hbs[0], ax=axes, fraction=0.018, pad=0.01)
cb.set_label("firing events per bin (log scale)", fontsize=10)
out = "/scratch/euh7ys/climate_xai/plots/concept_scatter_ivt_activation.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out, "| gmax", gmax)
