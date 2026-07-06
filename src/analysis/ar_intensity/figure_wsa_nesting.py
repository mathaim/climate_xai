"""Matryoshka nesting in W_S_America: broad Southern-Ocean parent 340 contains Chilean-coast children (with coastlines)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.patches import Rectangle
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/characterize_wsa.npz")
nlat, nlon = D["nlat"], D["nlon"]
def ev(cc): return D[f"n_{cc}"], D[f"a_{cc}"]
EXT = [-95, -55, -70, -22]; proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(8.5, 8)); ax = plt.axes(projection=proj)
ax.set_extent(EXT, crs=proj)
ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
pn, _ = ev(340)
H, xe, ye = np.histogram2d(nlon[pn], nlat[pn], bins=[np.linspace(EXT[0], EXT[1], 70), np.linspace(EXT[2], EXT[3], 90)])
pm = ax.pcolormesh(xe, ye, H.T, cmap="Blues", alpha=0.8, shading="auto", transform=proj, zorder=1)
fig.colorbar(pm, ax=ax, shrink=0.6, label="parent 340 firing frequency (broad Southern-Ocean storm track)")
ax.coastlines(resolution="110m", lw=0.9, color="k", zorder=5)
ax.add_feature(cfeature.BORDERS, lw=0.3, alpha=0.4, zorder=5)
for cc, col, lab in [(3481, "#d62728", "3481  S. Chile core (IVT 239, P=1.00)"),
                     (3948, "#ff7f0e", "3948  S-central Chile (~38S)"),
                     (3675, "#2ca02c", "3675  central Chile (~35S)")]:
    n, a = ev(cc); s = a >= np.quantile(a, 0.99)
    ax.scatter(nlon[n[s]], nlat[n[s]], s=20, c=col, edgecolor="k", lw=0.3, label=lab, transform=proj, zorder=6)
ax.add_patch(Rectangle((-77, -50), 15, 20, fill=False, edgecolor="k", lw=1.6, ls="--", transform=proj, zorder=7))
ax.text(-76.5, -31, "W_S_America box", fontsize=8, zorder=8)
gl = ax.gridlines(draw_labels=True, lw=0.3, color="0.6", alpha=0.5); gl.top_labels = gl.right_labels = False
ax.set_title("Matryoshka nesting in W_S_America: broad parent 340 contains Chilean-coast children")
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
fig.savefig("/scratch/euh7ys/climate_xai/plots/wsa_nesting.png", dpi=170, bbox_inches="tight"); print("saved")
