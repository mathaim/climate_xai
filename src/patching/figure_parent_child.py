"""Same-layer parent->child dependence: child code at landfall vs parent gain, mean +/- std over timesteps."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = np.load("/scratch/euh7ys/climate_xai/patching/parent_child_sae.npz")
gains = D["gains"]; gi1 = int(np.where(gains == 1.0)[0][0])
fig, ax = plt.subplots(figsize=(7.2, 5.2))
COL = {3481: "#d62728", 3948: "#ff7f0e", 3675: "#2ca02c"}
LAB = {3481: "child 3481 (~47S)", 3948: "child 3948 (~38S)", 3675: "child 3675 (~35S)"}
for cc in [3481, 3948, 3675]:
    cm = D[f"child_{cc}"]; ok = cm[:, gi1] > 1e-6; rel = cm[ok] / cm[ok, gi1:gi1+1]
    ax.errorbar(gains, rel.mean(0), yerr=rel.std(0), marker="o", capsize=3, color=COL[cc], label=LAB[cc])
pm = D["parent_3481"]; prel = pm / np.maximum(pm[:, gi1:gi1+1], 1e-6)
ax.plot(gains, prel.mean(0), "--", color="0.5", label="parent 340 (intervention level)")
ax.axhline(1, color="0.85", ls=":"); ax.axvline(1, color="0.85", ls=":")
ax.set_xlabel("parent 340 gain $g$   (0 clamp, 1 baseline, 2 amplify)")
ax.set_ylabel("code at landfall, relative to baseline")
ax.set_title("Same-layer parent$\\to$child dependence (mean $\\pm$ std over 6 timesteps per child)")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/parent_child_sae.png", dpi=170, bbox_inches="tight"); print("saved")
