"""AR vs non-AR classification: median firing IVT and %-at-AR-level vs the W_S_America thresholds."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
con = ["340", "1829", "3481", "99"]; medivt = [91, 121, 122, 331]; fp50 = [7, 9, 9, 51]
col = ["#7f8c8d", "#2980b9", "#27ae60", "#c0392b"]; p10, p50, p90 = 179, 319, 554
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].bar(con, medivt, color=col)
for th, lab in [(p10, "p10"), (p50, "p50 = AR"), (p90, "p90")]:
    ax[0].axhline(th, ls="--", color="0.5"); ax[0].text(3.35, th + 5, lab, fontsize=8, color="0.4")
ax[0].set_ylabel("median firing IVT (kg m$^{-1}$ s$^{-1}$)"); ax[0].set_title("Where each concept fires vs AR thresholds")
ax[1].bar(con, fp50, color=col); ax[1].axhline(50, ls=":", color="0.5")
ax[1].set_ylabel("% of firings at AR level ( > p50 )"); ax[1].set_title("AR-level firing fraction")
for i, v in enumerate(fp50): ax[1].text(i, v + 1, f"{v}%", ha="center", fontsize=9)
fig.suptitle("Only 99 is an AR concept; 340/1829/3481 fire at sub-AR moisture", y=1.02)
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/ar_classification.png", dpi=170, bbox_inches="tight"); print("saved classification")
