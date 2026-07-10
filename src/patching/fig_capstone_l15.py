"""Capstone L15-response figure from the reader outputs. (a) 99's L15 committee vs dose;
(b) 1592: successor 2251 sign-coherent, overlap pick 1226 incoherent, control 3163 at floor."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
# (a) 99 committee: gain 0 (clamp), 1 (baseline), 2, 3
g = [0, 1, 2, 3]
resp = {111: [-76.0, 0, 57.9, 82.0], 214: [-26.4, 0, 43.3, 92.5],
        123: [-25.9, 0, 12.0, 16.1], 864: [-6.2, 0, 10.0, 17.4]}
cols = {111: "#c0392b", 214: "#e67e22", 123: "#f1c40f", 864: "#7f8c8d"}
for t, v in resp.items():
    ax[0].plot(g, v, "-o", ms=5, color=cols[t], label=f"L15 {t}")
ax[0].axhline(0, color="0.7", lw=0.8); ax[0].axvline(1, color="0.85", lw=0.8, ls=":")
ax[0].set_xticks(g); ax[0].set_xticklabels(["clamp\n(g=0)", "baseline\n(g=1)", "amplify\n(g=2)", "amplify\n(g=3)"], fontsize=8)
ax[0].set_ylabel("total L15 code change vs baseline"); ax[0].set_title("(a) concept 99: layer-15 committee follows the layer-8 dial", fontsize=10, loc="left")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
# (b) 1592 verdict: clamp vs inject per target
targets = ["2251\nsuccessor", "1226\noverlap pick", "3163\ncontrol"]
clamp = [-15.9, -7.8, 0.5]; inject = [187.1, -135.2, -1.1]
x = np.arange(3); w = 0.36
ax[1].bar(x - w/2, clamp, w, color="#2980b9", label="clamp during AR")
ax[1].bar(x + w/2, inject, w, color="#c0392b", label="inject into clear air")
ax[1].axhline(0, color="0.7", lw=0.8)
ax[1].set_xticks(x); ax[1].set_xticklabels(targets, fontsize=8.5)
ax[1].set_ylabel("total L15 code change vs baseline")
ax[1].set_title("(b) concept 1592: function-selected successor responds coherently", fontsize=10, loc="left")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.25, axis="y")
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/capstone_l15.png", dpi=170, bbox_inches="tight")
print("saved capstone_l15.png")
