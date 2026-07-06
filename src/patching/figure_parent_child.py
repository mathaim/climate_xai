"""Same-layer parent->child dependence: children on a zoomed left axis, parent level on right axis."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = np.load("/scratch/euh7ys/climate_xai/patching/parent_child_sae.npz")
gains = D["gains"]; gi1 = int(np.where(gains == 1.0)[0][0])
fig, ax = plt.subplots(figsize=(7.6, 5.3))
COL = {3481: "#d62728", 3948: "#ff7f0e", 3675: "#2ca02c"}
LAB = {3481: "child 3481 (~47S)", 3948: "child 3948 (~38S)", 3675: "child 3675 (~35S)"}
for cc in [3481, 3948, 3675]:
    cm = D[f"child_{cc}"]; ok = cm[:, gi1] > 1e-6; rel = cm[ok] / cm[ok, gi1:gi1+1]
    ax.errorbar(gains, rel.mean(0), yerr=rel.std(0), marker="o", ms=6, lw=2, capsize=3, color=COL[cc], label=LAB[cc])
ax.set_ylim(0.78, 1.22); ax.axhline(1, color="0.85", ls=":"); ax.axvline(1, color="0.85", ls=":")
ax.set_xlabel("parent 340 gain $g$   (0 clamp, 1 baseline, 2 amplify)")
ax.set_ylabel("child code at landfall, relative to baseline")
ax2 = ax.twinx()
pm = D["parent_3481"]; prel = pm / np.maximum(pm[:, gi1:gi1+1], 1e-6)
ax2.plot(gains, prel.mean(0), "--", color="0.55", lw=1.6, label="parent 340 level (right axis)")
ax2.set_ylim(0, 2.1); ax2.set_ylabel("parent 340 code, relative to baseline", color="0.45")
ax2.tick_params(axis="y", labelcolor="0.45")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
ax.set_title("Same-layer parent$\\to$child dependence (mean $\\pm$ std over 6 timesteps per child)")
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/parent_child_sae.png", dpi=170, bbox_inches="tight"); print("saved")
