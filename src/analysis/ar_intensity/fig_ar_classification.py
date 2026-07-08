"""AR vs coastal-moisture classification for all 8 core concepts (both parents + all children):
median firing IVT (left, vs standard AR thresholds) and fraction of firings at AR / strong-AR
strength (right). No in-image title."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
con   = ["340","3481","3948","3675","99","1454","3392","2722"]
med   = [78, 130, 89, 68, 430, 534, 689, 385]
ar250 = [11, 19, 9, 6, 77, 85, 96, 71]
ar500 = [1, 2, 0, 1, 41, 54, 73, 33]
col   = ["#2980b9"]*4 + ["#c0392b"]*4
x = np.arange(len(con))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
# left: median firing IVT vs AR thresholds
ax[0].bar(x, med, color=col)
for th, lab in [(250, "AR ($\\geq$250)"), (500, "strong AR ($\\geq$500)")]:
    ax[0].axhline(th, ls="--", color="0.5", lw=1); ax[0].text(len(con)-0.45, th+10, lab, fontsize=8, color="0.4", ha="right")
ax[0].set_xticks(x); ax[0].set_xticklabels(con); ax[0].set_ylim(0, 780)
ax[0].set_ylabel("median firing IVT (kg m$^{-1}$s$^{-1}$)")
# right: fraction of firings at AR and strong-AR strength
w = 0.4
ax[1].bar(x - w/2, ar250, w, color=col, label="AR ($\\geq$250)")
ax[1].bar(x + w/2, ar500, w, color=col, alpha=0.45, label="strong ($\\geq$500)")
ax[1].set_xticks(x); ax[1].set_xticklabels(con); ax[1].set_ylim(0, 105)
ax[1].set_ylabel("% of firings"); ax[1].legend(fontsize=8, loc="upper left")
for a in ax:
    a.axvspan(-0.5, 3.5, color="#2980b9", alpha=0.05); a.axvspan(3.5, 7.5, color="#c0392b", alpha=0.05)
ax[0].text(1.5, 735, "geographic (coastal moisture)", ha="center", color="#2980b9", fontsize=9)
ax[0].text(5.5, 735, "AR-intensity", ha="center", color="#c0392b", fontsize=9)
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/ar_classification.png", dpi=170, bbox_inches="tight")
print("saved ar_classification.png")
