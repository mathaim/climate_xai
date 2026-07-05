"""3x2 stacked snapshot grid: rows = peak timestamps, cols = IVT+transport | concept 1592 activation."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.analysis.ar_intensity.regions import REGIONS
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/snapshots_1592.npz")
labels = [str(x) for x in D["labels"]]; nlat = D["nlat"]; nlon = D["nlon"]; n = len(labels)
R = "W_N_America"; cfg = REGIONS[R]; la = cfg["lat"]; lons = cfg["lon"]; conv = lambda x: x - 360 if x > 180 else x
xs = [conv(x) for seg in lons for x in seg]; ext = [min(xs) - 15, max(xs) + 15, la[0] - 10, la[1] + 12]; PC = ccrs.PlateCarree()
CMAP = LinearSegmentedColormap.from_list("moist", ["#e8f6f9", "#9fdcc4", "#4eb3d3", "#2b6cb0", "#3b3b98", "#7d3c98", "#c0392b"])
ie = (nlon >= ext[0]) & (nlon <= ext[1]) & (nlat >= ext[2]) & (nlat <= ext[3])
av = np.concatenate([D[f"val{i}"][ie] for i in range(n)]); vz = float(np.percentile(av[av > 0], 99))
IVTLEV = [100, 250, 400, 550, 700, 850, 1000, 1200]; ACTLEV = np.linspace(vz * .05, vz, 12)
sub = np.where(ie)[0]; sub = sub[::max(1, len(sub) // 350)]
fig = plt.figure(figsize=(11, 4.8 * n)); gs = fig.add_gridspec(n, 2, wspace=0.04, hspace=0.04)
def geo(k, col):
    A = fig.add_subplot(gs[k, col], projection=PC); A.add_feature(cfeature.LAND, facecolor="#f2efe9", zorder=0)
    A.add_feature(cfeature.OCEAN, facecolor="#e3edf3", zorder=0); A.coastlines("50m", color="#555", lw=.5, zorder=5); A.set_extent(ext, crs=PC); return A
def box(A):
    for x0, x1 in lons: A.plot([conv(x0), conv(x1), conv(x1), conv(x0), conv(x0)], [la[0], la[0], la[1], la[1], la[0]], c="#c0392b", lw=1.6, transform=PC, zorder=8)
ax0, ax1 = [], []
for i in range(n):
    mag = D[f"mag{i}"]; qu = D[f"qu{i}"]; qv = D[f"qv{i}"]; val = D[f"val{i}"]
    A = geo(i, 0)
    cf = A.tricontourf(nlon, nlat, mag, levels=IVTLEV, cmap="YlGnBu", extend="max", transform=PC, zorder=2)
    A.tricontour(nlon, nlat, mag, [250], colors="k", linewidths=1.1, transform=PC, zorder=4)
    A.quiver(nlon[sub], nlat[sub], qu[sub], qv[sub], transform=PC, scale=2.2e4, width=.0035, color="#222", zorder=6)
    box(A); A.gridlines(lw=.2, color="#ccc")
    A.text(-0.07, 0.5, labels[i], transform=A.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold", clip_on=False)
    B = geo(i, 1)
    mp = B.tricontourf(nlon, nlat, np.clip(val, 0, vz), levels=ACTLEV, cmap=CMAP, extend="max", transform=PC, zorder=2)
    B.tricontour(nlon, nlat, mag, [250], colors="k", linewidths=.9, linestyles="--", transform=PC, zorder=4)
    box(B); B.gridlines(lw=.2, color="#ccc"); ax0.append(A); ax1.append(B)
    if i == 0:
        A.set_title("IVT + moisture transport", fontsize=14); B.set_title("SAE concept 1592 activation", fontsize=14)
fig.colorbar(cf, ax=ax0, location="bottom", shrink=0.7, pad=0.02, label="IVT kg m$^{-1}$ s$^{-1}$")
fig.colorbar(mp, ax=ax1, location="bottom", shrink=0.7, pad=0.02, label="activation")
out = "/scratch/euh7ys/climate_xai/plots/snapshots_1592_grid.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
