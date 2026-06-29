"""Per-concept scatter of firing events: x = region IVT, y = concept activation, colour = region.
Concepts 99 (general intensity), 3153 (extreme E. Australia), 3483 (Northern Hemisphere). Matryoshka L8."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"
REGIONS = [("W_N_America", "W.N.Am", "#d62728"), ("W_Europe", "W.Eu", "#1f77b4"),
           ("W_S_America", "W.S.Am", "#2ca02c"), ("E_Australia", "E.Aus", "#9467bd")]
CONCEPTS = [(99, "Concept 99 \u2014 general intensity"),
            (3153, "Concept 3153 \u2014 extreme E. Australia"),
            (3483, "Concept 3483 \u2014 Northern Hemisphere")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
for ax, (c, title) in zip(axes, CONCEPTS):
    series = []
    for rkey, rlab, col in REGIONS:
        try:
            t = np.load(f"{TRACK}/track_matry_{rkey}.npz")
        except FileNotFoundError:
            print("missing", rkey); continue
        A = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float)
        ok = np.isfinite(ivt) & (A > 0)
        series.append((rlab, col, ivt[ok], A[ok]))
        print(f"concept {c:5d}  {rlab:7s}  firing events {int(ok.sum())}")
    for rlab, col, x, y in sorted(series, key=lambda s: -len(s[2])):   # dense region first
        ax.scatter(x, y, s=7, c=col, alpha=0.3, edgecolor="none")
    ax.set_title(title, fontsize=11); ax.set_xlabel("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=10)
axes[0].set_ylabel("concept activation", fontsize=10)
handles = [Line2D([], [], marker="o", ls="", color=col, label=rlab, ms=7) for _, rlab, col in REGIONS]
axes[0].legend(handles=handles, title="region", fontsize=9, loc="upper left", framealpha=0.9)
fig.tight_layout()
out = "/scratch/euh7ys/climate_xai/plots/concept_scatter_ivt_activation.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
