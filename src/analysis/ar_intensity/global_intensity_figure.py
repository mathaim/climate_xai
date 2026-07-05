"""Keyhole-free intensity dose-response: mean activation vs local node IVT (all mesh nodes)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/global_intensity.npz")
dose, bins, cnt, concepts = D["dose"], D["bins"], D["cnt"], [int(x) for x in D["concepts"]]
ctr = (bins[:-1] + bins[1:]) / 2
LAB = {99: "99 parent (general)", 3153: "3153 child (E.Aus extreme)", 3483: "3483 child (NH)"}
COL = {99: "#c0392b", 3153: "#9467bd", 3483: "#1f77b4"}
print("IVT bin centers:", [int(x) for x in ctr])
for k, cc in enumerate(concepts):
    print(f"{cc:>5}:", " ".join(f"{v:.3f}" for v in dose[k]))
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for k, cc in enumerate(concepts):
    ax[0].plot(ctr, dose[k], "-o", ms=4, color=COL.get(cc), label=LAB.get(cc, str(cc)))
    ax[1].plot(ctr, dose[k] / dose[k].max(), "-o", ms=4, color=COL.get(cc), label=LAB.get(cc, str(cc)))
ax[0].set_ylabel("mean activation (all mesh nodes)"); ax[0].set_title("raw")
ax[1].set_ylabel("normalized to own max"); ax[1].set_title("shape")
for a in ax: a.set_xlabel("local node IVT (kg m$^{-1}$ s$^{-1}$)"); a.legend(fontsize=9); a.grid(alpha=.3)
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/global_intensity_doseresponse.png", dpi=170, bbox_inches="tight"); print("saved")
