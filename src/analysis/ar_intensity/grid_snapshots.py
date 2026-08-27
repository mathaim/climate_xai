"""2x3 snapshot grid: rows = IVT+transport | concept 1592 activation, cols = peak timestamps."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.analysis.ar_intensity.regions import REGIONS
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/snapshots_1592.npz")
def _fmt(x):
    p=str(x).replace("z","").replace("T"," ").strip().split(); return p[0]+"T"+((p[1][:2]) if len(p)>1 else "00")+"-00"
labels = [_fmt(x) for x in D["labels"]]; nlat = D["nlat"]; nlon = D["nlon"]; n = len(labels)
R = "W_N_America"; cfg = REGIONS[R]; la = cfg["lat"]; lons = cfg["lon"]; conv = lambda x: x - 360 if x > 180 else x
xs = [conv(x) for seg in lons for x in seg]; ext = [min(xs) - 15, max(xs) + 15, la[0] - 10, la[1] + 12]; PC = ccrs.PlateCarree()
CMAP = LinearSegmentedColormap.from_list("moist", ["#e8f6f9", "#9fdcc4", "#4eb3d3", "#2b6cb0", "#3b3b98", "#7d3c98", "#c0392b"])
ie = (nlon >= ext[0]) & (nlon <= ext[1]) & (nlat >= ext[2]) & (nlat <= ext[3])
av = np.concatenate([D[f"val{i}"][ie] for i in range(n)]); vz = 0.5
IVTLEV = [100, 250, 400, 550, 700, 850, 1000, 1200]; ACTLEV = np.linspace(vz * .05, vz, 12)
sub = np.where(ie)[0]; sub = sub[::max(1, len(sub) // 350)]
fig = plt.figure(figsize=(4.6 * n, 7.9)); gs = fig.add_gridspec(2, n, wspace=0.03, hspace=0.01)
def geo(row, col):
    A = fig.add_subplot(gs[row, col], projection=PC); A.add_feature(cfeature.LAND, facecolor="#f2efe9", zorder=0)
    A.add_feature(cfeature.OCEAN, facecolor="#e3edf3", zorder=0); A.coastlines("50m", color="#555", lw=.5, zorder=5); A.set_extent(ext, crs=PC); return A
def box(A):
    for x0, x1 in lons: A.plot([conv(x0), conv(x1), conv(x1), conv(x0), conv(x0)], [la[0], la[0], la[1], la[1], la[0]], c="#c0392b", lw=1.6, transform=PC, zorder=8)
axI, axA = [], []
for i in range(n):
    mag = D[f"mag{i}"]; qu = D[f"qu{i}"]; qv = D[f"qv{i}"]; val = D[f"val{i}"]
    A = geo(0, i)
    cf = A.tricontourf(nlon, nlat, mag, levels=IVTLEV, cmap="YlGnBu", extend="max", transform=PC, zorder=2)
    A.tricontour(nlon, nlat, mag, [250], colors="k", linewidths=1.1, transform=PC, zorder=4)
    A.quiver(nlon[sub], nlat[sub], qu[sub], qv[sub], transform=PC, scale=2.2e4, width=.0035, color="#222", zorder=6)
    box(A); A.gridlines(lw=.2, color="#ccc"); A.set_title(labels[i], fontsize=13, fontweight="bold")
    B = geo(1, i)
    mp = B.tricontourf(nlon, nlat, np.clip(val, 0, vz), levels=ACTLEV, cmap=CMAP, extend="max", transform=PC, zorder=2)
    B.tricontour(nlon, nlat, mag, [250], colors="k", linewidths=.9, linestyles="--", transform=PC, zorder=4)
    box(B); B.gridlines(lw=.2, color="#ccc"); axI.append(A); axA.append(B)
    if i == 0:
        A.text(-0.09, 0.5, "IVT + moisture transport", transform=A.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold", clip_on=False)
        B.text(-0.09, 0.5, "SAE concept 1592 activation", transform=B.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold", clip_on=False)
fig.colorbar(cf, ax=axI, location="right", shrink=0.85, pad=0.015, label="IVT kg m$^{-1}$ s$^{-1}$")
cbA = fig.colorbar(mp, ax=axA, location="right", shrink=0.85, pad=0.015, label="activation"); cbA.set_ticks([0.1,0.2,0.3,0.4,0.5])
out = "/scratch/euh7ys/climate_xai/plots/snapshots_1592_grid.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
