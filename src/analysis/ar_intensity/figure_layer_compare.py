"""L8 vs L15 comparison of the WSA nesting concepts: children stay put, broad parent fragments."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.patches import Rectangle
D8 = np.load("/scratch/euh7ys/climate_xai/concept_ivt/characterize_wsa.npz")
D15 = np.load("/scratch/euh7ys/climate_xai/concept_ivt/characterize_l15.npz")
nlat, nlon = D8["nlat"], D8["nlon"]; EXT = [-95, -55, -70, -22]; proj = ccrs.PlateCarree()
def strong(D, cc):
    n, a = D[f"n_{cc}"], D[f"a_{cc}"]; s = a >= np.quantile(a, 0.99); return nlon[n[s]], nlat[n[s]]
fig, axes = plt.subplots(1, 2, figsize=(15, 8), subplot_kw={"projection": proj})
LAB = ["S. Chile ~47S", "S-central ~38S", "central ~35S"]; COL = ["#d62728", "#ff7f0e", "#2ca02c"]
CH8 = [3481, 3948, 3675]; CH15 = [3160, 3392, 1980]
def base(ax, title):
    ax.set_extent(EXT, crs=proj); ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
    ax.coastlines("110m", lw=0.9, zorder=4)
    ax.add_patch(Rectangle((-77, -50), 15, 20, fill=False, edgecolor="k", lw=1.3, ls="--", transform=proj, zorder=6))
    ax.set_title(title)
ax = axes[0]; base(ax, "L8")
n = D8["n_340"]; H, xe, ye = np.histogram2d(nlon[n], nlat[n], bins=[np.linspace(EXT[0], EXT[1], 70), np.linspace(EXT[2], EXT[3], 90)])
pm = ax.pcolormesh(xe, ye, H.T, cmap="Blues", alpha=0.7, shading="auto", transform=proj, zorder=1)
fig.colorbar(pm, ax=ax, shrink=0.55, pad=0.02, label="340 (parent) firing frequency")
for cc, col, lab in zip(CH8, COL, LAB):
    lo, la = strong(D8, cc); ax.scatter(lo, la, s=20, c=col, edgecolor="k", lw=0.3, transform=proj, zorder=5, label=f"{cc}  {lab}")
ax.legend(fontsize=8, loc="upper left")
ax = axes[1]; base(ax, "L15")
nL15 = np.concatenate([D15[f"n_{pc}"] for pc in [1536, 1675, 756]])
H15, xe15, ye15 = np.histogram2d(nlon[nL15], nlat[nL15], bins=[np.linspace(EXT[0], EXT[1], 70), np.linspace(EXT[2], EXT[3], 90)])
pm15 = ax.pcolormesh(xe15, ye15, H15.T, cmap="Blues", alpha=0.7, shading="auto", transform=proj, zorder=1)
fig.colorbar(pm15, ax=ax, shrink=0.55, pad=0.02, label="340 fragments (1536,1675,756) firing frequency")
for cc, col, lab in zip(CH15, COL, LAB):
    lo, la = strong(D15, cc); ax.scatter(lo, la, s=20, c=col, edgecolor="k", lw=0.3, transform=proj, zorder=5, label=f"{cc}  {lab}")
ax.legend(fontsize=8, loc="upper left")
fig.savefig("/scratch/euh7ys/climate_xai/plots/layer_compare.png", dpi=170, bbox_inches="tight"); print("saved")
