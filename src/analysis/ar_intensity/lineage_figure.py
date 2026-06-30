"""Stacked global footprints of the G0->G4 lineage: 99 -> 411 -> 941 -> 1838 -> 3153."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/global_footprint_lineage.npz")
foot, nlat, nlon, concepts = D["foot"], D["nlat"], D["nlon"], D["concepts"]
lon = np.where(nlon > 180, nlon - 360, nlon)
LAB = {99:"G0  concept 99  -  general intensity (global)",
       411:"G1  concept 411  -  E. Australia, austral summer",
       941:"G2  concept 941  -  E. Australia, February",
       1838:"G3  concept 1838  -  E. Australia, February (narrower)",
       3153:"G4  concept 3153  -  E. Australia extreme summer (La Nina)"}
PC = ccrs.PlateCarree(); n = len(concepts)
fig, axes = plt.subplots(n, 1, figsize=(11, 3.1 * n), subplot_kw=dict(projection=ccrs.Robinson()))
for ax, k in zip(np.atleast_1d(axes), range(n)):
    cc = int(concepts[k]); v = foot[k]
    ax.set_global(); ax.add_feature(cfeature.LAND, facecolor="#efece6", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#e6eef4", zorder=0); ax.coastlines("110m", color="#666", lw=.4)
    sc = ax.scatter(lon, nlat, c=v, s=6, cmap="magma", vmin=0, vmax=np.percentile(v, 99.5), transform=PC, edgecolor="none", zorder=3)
    ax.set_title(LAB.get(cc, str(cc)), fontsize=12, loc="left")
    fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.02, label="mean DJF activation")
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/lineage_footprints.png", dpi=160, bbox_inches="tight"); print("saved")
