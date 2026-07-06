"""Matryoshka nesting map: broad monsoon parent 230 (density) contains localized intense children."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/characterize.npz")
nlat, nlon = D["nlat"], D["nlon"]
def ev(cc): return D[f"n_{cc}"], D[f"a_{cc}"]
EXT = [30, 115, -15, 32]
inx = (nlon >= EXT[0]) & (nlon <= EXT[1]) & (nlat >= EXT[2]) & (nlat <= EXT[3])
fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(nlon[inx], nlat[inx], s=1, c="0.88", zorder=0)
pn, _ = ev(230)
H, xe, ye = np.histogram2d(nlon[pn], nlat[pn], bins=[np.linspace(EXT[0], EXT[1], 90), np.linspace(EXT[2], EXT[3], 60)])
pm = ax.pcolormesh(xe, ye, H.T, cmap="Blues", alpha=0.75, shading="auto", zorder=1)
fig.colorbar(pm, ax=ax, shrink=0.7, label="parent 230 firing frequency (broad monsoon)")
for cc, col, lab in [(4094, "#d62728", "4094  Arabian Sea core (P=1.00, IVT 636)"),
                     (1986, "#ff7f0e", "1986  Somali/Findlater jet"),
                     (3167, "#2ca02c", "3167  Bay of Bengal / S. India")]:
    n, a = ev(cc); s = a >= np.quantile(a, 0.99)
    ax.scatter(nlon[n[s]], nlat[n[s]], s=16, c=col, edgecolor="k", lw=0.2, label=lab, zorder=3)
ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:]); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Matryoshka nesting: broad SW-monsoon parent (230) contains localized intense children (JJA)")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
fig.savefig("/scratch/euh7ys/climate_xai/plots/monsoon_nesting.png", dpi=170, bbox_inches="tight"); print("saved")
