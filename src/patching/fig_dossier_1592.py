"""1592 dossier: successor 2251's firing level and the regional forecast, across the four
conditions (AR day / AR day with 1592 removed / clear day / clear day with 1592 injected)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
conds = ["AR day", "AR day,\n1592 removed", "clear day", "clear day,\n1592 injected"]
lvl_2251 = [45.9, 30.0, 46.1, 233.3]
box_mean = [178.2, 161.2, 115.6, 312.0]
cols = ["#5b8db8", "#2980b9", "#c9a66b", "#c0392b"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(conds, lvl_2251, color=cols)
ax[0].set_ylabel("total firing of layer-15 successor 2251")
ax[0].set_title("(a) the successor follows the concept", fontsize=10, loc="left")
for i, v in enumerate(lvl_2251): ax[0].text(i, v + 4, f"{v:.0f}", ha="center", fontsize=9)
ax[1].bar(conds, box_mean, color=cols)
ax[1].set_ylabel("W. N. America box mean IVT (kg m$^{-1}$ s$^{-1}$)")
ax[1].set_title("(b) the forecast follows the concept", fontsize=10, loc="left")
for i, v in enumerate(box_mean): ax[1].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9)
for a in ax: a.grid(alpha=.25, axis="y"); a.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/dossier_1592.png", dpi=170, bbox_inches="tight")
print("saved dossier_1592.png")
