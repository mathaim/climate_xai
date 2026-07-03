"""Parent 512 (SH moisture) and child 1308 (intense E. Australia, fires only when 512 fires)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.analysis.ar_intensity.regions import REGIONS
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/global_footprint_pair.npz")
foot, mclim, nlat, nlon, concepts = D["foot"], D["monthclim"], D["nlat"], D["nlon"], D["concepts"]
lon = np.where(nlon > 180, nlon - 360, nlon); MON = "J F M A M J J A S O N D".split()
LAB = {512: "Parent 512  -  Southern Hemisphere moisture",
       1308: "Child 1308  -  intense E. Australia  (fires only when 512 fires)"}
cv = lambda x: x - 360 if x > 180 else x
PC = ccrs.PlateCarree(); n = len(concepts); LET = "abcdefgh"
fig = plt.figure(figsize=(14, 3.2 * n))
gs = fig.add_gridspec(n, 2, width_ratios=[3.3, 1], wspace=0.10, hspace=0.30)
for k in range(n):
    cc = int(concepts[k]); v = foot[k]
    ax = fig.add_subplot(gs[k, 0], projection=ccrs.Robinson())
    ax.set_global(); ax.add_feature(cfeature.LAND, facecolor="#d9d2c5", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f4", zorder=0); ax.coastlines("110m", color="#333333", lw=.7)
    ax.add_feature(cfeature.BORDERS, lw=.3, edgecolor="#888"); ax.gridlines(lw=.3, color="#aaa", alpha=.6)
    for _r, _c in REGIONS.items():
        _la = _c["lat"]
        for _x0, _x1 in _c["lon"]:
            ax.plot([cv(_x0), cv(_x1), cv(_x1), cv(_x0), cv(_x0)], [_la[0], _la[0], _la[1], _la[1], _la[0]],
                    c="#1f6f8b", lw=1.0, transform=PC, zorder=5)
    vmx = max(np.percentile(v, 99.5), 1e-6); msk = v > 0.02 * vmx
    sc = ax.scatter(lon[msk], nlat[msk], c=v[msk], s=7, cmap="YlOrBr", vmin=0, vmax=vmx,
                    transform=PC, edgecolor="none", zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    ax.text(0.0, 1.04, f"({LET[2*k]})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    axm = fig.add_subplot(gs[k, 1]); axm.bar(range(12), mclim[k], color="#8c6d3f")
    axm.set_xticks(range(12)); axm.set_xticklabels(MON, fontsize=7)
    axm.tick_params(labelsize=7)
    axm.text(-0.14, 1.06, f"({LET[2*k+1]})", transform=axm.transAxes, fontsize=13, fontweight="bold", va="bottom")
fig.savefig("/scratch/euh7ys/climate_xai/plots/pair_512_1308.png", dpi=150, bbox_inches="tight"); print("saved")
