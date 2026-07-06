"""Matryoshka nesting in W_S_America: broad Southern-Ocean parent 340 contains Chilean-coast children."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/characterize_wsa.npz")
nlat, nlon = D["nlat"], D["nlon"]
def ev(cc): return D[f"n_{cc}"], D[f"a_{cc}"]
EXT = [-95, -55, -70, -22]
inx = (nlon >= EXT[0]) & (nlon <= EXT[1]) & (nlat >= EXT[2]) & (nlat <= EXT[3])
fig, ax = plt.subplots(figsize=(8.5, 8))
ax.scatter(nlon[inx], nlat[inx], s=2, c="0.88", zorder=0)
pn, _ = ev(340)
H, xe, ye = np.histogram2d(nlon[pn], nlat[pn], bins=[np.linspace(EXT[0], EXT[1], 70), np.linspace(EXT[2], EXT[3], 90)])
pm = ax.pcolormesh(xe, ye, H.T, cmap="Blues", alpha=0.75, shading="auto", zorder=1)
fig.colorbar(pm, ax=ax, shrink=0.6, label="parent 340 firing frequency (broad Southern-Ocean storm track)")
for cc, col, lab in [(3481, "#d62728", "3481  S. Chile core (IVT 239, P=1.00)"),
                     (3948, "#ff7f0e", "3948  S-central Chile (~38S)"),
                     (3675, "#2ca02c", "3675  central Chile (~35S)")]:
    n, a = ev(cc); s = a >= np.quantile(a, 0.99)
    ax.scatter(nlon[n[s]], nlat[n[s]], s=18, c=col, edgecolor="k", lw=0.3, label=lab, zorder=3)
ax.add_patch(Rectangle((-77, -50), 15, 20, fill=False, edgecolor="k", lw=1.6, ls="--", zorder=4))
ax.text(-76.5, -31, "W_S_America box", fontsize=8, zorder=5)
ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:]); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Matryoshka nesting in W_S_America: broad parent 340 contains Chilean-coast children")
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
fig.savefig("/scratch/euh7ys/climate_xai/plots/wsa_nesting.png", dpi=170, bbox_inches="tight"); print("saved")
