"""Per-concept scatter of firing events: x = region IVT, y = concept activation, colour = region.
Equal random sample per region so all four stay visible. Matryoshka L8 concepts 99 / 3153 / 3483."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; NCAP = 4000; rng = np.random.default_rng(0)
REGIONS = [("W_N_America", "W.N.Am", "#d62728"), ("W_Europe", "W.Eu", "#1f77b4"),
           ("W_S_America", "W.S.Am", "#2ca02c"), ("E_Australia", "E.Aus", "#9467bd")]
CONCEPTS = [(99, "Concept 99 \u2014 general intensity"),
            (3153, "Concept 3153 \u2014 extreme E. Australia"),
            (3483, "Concept 3483 \u2014 Northern Hemisphere")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
for ax, (c, title) in zip(axes, CONCEPTS):
    xs, ys, cs = [], [], []
    for rkey, rlab, col in REGIONS:
        t = np.load(f"{TRACK}/track_matry_{rkey}.npz")
        A = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float)
        ok = np.isfinite(ivt) & (A > 0); x, y = ivt[ok], A[ok]
        if len(x) > NCAP:
            i = rng.choice(len(x), NCAP, replace=False); x, y = x[i], y[i]
        xs.append(x); ys.append(y); cs += [col] * len(x)
    X = np.concatenate(xs); Y = np.concatenate(ys); C = np.array(cs)
    p = rng.permutation(len(X))                                   # interleave colours
    ax.scatter(X[p], Y[p], s=6, c=C[p], alpha=0.4, edgecolor="none")
    ax.set_title(title, fontsize=11); ax.set_xlabel("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=10)
axes[0].set_ylabel("concept activation", fontsize=10)
handles = [Line2D([], [], marker="o", ls="", color=col, label=rlab, ms=7) for _, rlab, col in REGIONS]
axes[0].legend(handles=handles, title="region", fontsize=9, loc="upper left", framealpha=0.95)
fig.tight_layout()
out = "/scratch/euh7ys/climate_xai/plots/concept_scatter_ivt_activation.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
