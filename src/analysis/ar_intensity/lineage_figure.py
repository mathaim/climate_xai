"""G0->G4 lineage: per-node firing-frequency footprint (earth-toned) + monthly firing climatology."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/global_footprint_lineage.npz")
foot, mclim, nlat, nlon, concepts = D["foot"], D["monthclim"], D["nlat"], D["nlon"], D["concepts"]
lon = np.where(nlon > 180, nlon - 360, nlon); MON = "J F M A M J J A S O N D".split()
LAB = {99:"G0  concept 99", 411:"G1  concept 411", 941:"G2  concept 941",
       1838:"G3  concept 1838", 3153:"G4  concept 3153"}
from src.analysis.ar_intensity.regions import REGIONS
_cv = lambda x: x - 360 if x > 180 else x
PC = ccrs.PlateCarree(); n = len(concepts)
fig = plt.figure(figsize=(14, 3.0 * n))
gs = fig.add_gridspec(n, 2, width_ratios=[3.3, 1], wspace=0.10, hspace=0.32)
for k in range(n):
    cc = int(concepts[k]); v = foot[k]
    ax = fig.add_subplot(gs[k, 0], projection=ccrs.Robinson())
    ax.set_global(); ax.add_feature(cfeature.LAND, facecolor="#d9d2c5", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f4", zorder=0); ax.coastlines("110m", color="#7a7060", lw=.4)
    ax.gridlines(lw=0.3, color="#aaa", alpha=0.6)
    for _r, _c in REGIONS.items():
        _la = _c["lat"]
        for _x0, _x1 in _c["lon"]:
            ax.plot([_cv(_x0), _cv(_x1), _cv(_x1), _cv(_x0), _cv(_x0)],
                    [_la[0], _la[0], _la[1], _la[1], _la[0]], c="#1f6f8b", lw=1.1,
                    transform=PC, zorder=5)
    sc = ax.scatter(lon, nlat, c=v, s=6, cmap="YlOrBr", vmin=0, vmax=max(np.percentile(v, 99.5), 1e-6),
                    transform=PC, edgecolor="none", zorder=3)
    ax.set_title(LAB.get(cc, str(cc)), fontsize=11, loc="left")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label="firing frequency")
    axm = fig.add_subplot(gs[k, 1])
    axm.bar(range(12), mclim[k], color="#8c6d3f"); axm.set_xticks(range(12)); axm.set_xticklabels(MON, fontsize=7)
    axm.set_title("firing freq by month", fontsize=8.5); axm.tick_params(labelsize=7)
fig.savefig("/scratch/euh7ys/climate_xai/plots/lineage_footprints.png", dpi=150, bbox_inches="tight"); print("saved")
