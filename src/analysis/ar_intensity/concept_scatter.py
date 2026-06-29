"""Small-multiples density: each concept (row) x region (col), 2D density of IVT vs activation.
Reads the regional specialisation off which cells are populated. Matryoshka L8: 99 / 3153 / 3483."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"
REGIONS = [("W_N_America", "W.N.Am"), ("W_Europe", "W.Eu"), ("W_S_America", "W.S.Am"), ("E_Australia", "E.Aus")]
CONCEPTS = [(99, "Concept 99\ngeneral intensity"), (3153, "Concept 3153\nextreme E. Australia"),
            (3483, "Concept 3483\nN. Hemisphere")]
data = {}
for c, _ in CONCEPTS:
    for rkey, _ in REGIONS:
        t = np.load(f"{TRACK}/track_matry_{rkey}.npz")
        A = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float)
        ok = np.isfinite(ivt) & (A > 0); data[(c, rkey)] = (ivt[ok], A[ok])
fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey="row")
XMAX = 2800
for i, (c, clab) in enumerate(CONCEPTS):
    allA = np.concatenate([data[(c, r)][1] for r, _ in REGIONS]); ymax = np.percentile(allA, 99.5)
    for j, (rkey, rlab) in enumerate(REGIONS):
        ax = axes[i, j]; x, y = data[(c, rkey)]
        if len(x):
            ax.hexbin(x, y, gridsize=28, cmap="viridis", bins="log", mincnt=1, extent=(0, XMAX, 0, ymax))
        ax.text(0.96, 0.93, f"n={len(x):,}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#222")
        ax.set_xlim(0, XMAX); ax.set_ylim(0, ymax)
        if i == 0: ax.set_title(rlab, fontsize=12)
        if j == 0: ax.set_ylabel(f"{clab}\n\nactivation", fontsize=9.5)
        if i == 2: ax.set_xlabel("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=9.5)
fig.tight_layout()
out = "/scratch/euh7ys/climate_xai/plots/concept_scatter_ivt_activation.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out)
