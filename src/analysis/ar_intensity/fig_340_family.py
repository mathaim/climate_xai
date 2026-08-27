"""Concept-340 family paper figure (fig:340family).

A: every node where matry-L8 concept 340 fires over 8,000 sampled timesteps,
   shaded by active-timestep count (log). B: south-polar map of each child's
   90%-of-firing anchor nodes (56 distinct, 11 shared ringed black).
In : /scratch/euh7ys/climate_xai/concept_ivt/footprint_340_family_8k.npz
Out: /scratch/euh7ys/climate_xai/plots/footprint_340_family.png
Run: cd ~/climate_xai && python src/analysis/ar_intensity/fig_340_family.py
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

NPZ = "/scratch/euh7ys/climate_xai/concept_ivt/footprint_340_family_8k.npz"
OUT = "/scratch/euh7ys/climate_xai/plots/footprint_340_family.png"
LAND = "#efe9dc"; SEA = "#dbe7f3"
SAM  = [2858, 3112, 3481, 3675, 3700, 1622, 3948]   # Chilean/Andean coast
APEN = [3757, 2474, 3495, 1399]                     # Antarctic Peninsula
NZ   = [3126]                                       # New Zealand
CHILDREN = SAM + APEN + NZ
blues = plt.get_cmap("Blues"); reds = plt.get_cmap("Reds")
COLOR = {}
for i, c in enumerate(SAM):  COLOR[c] = blues(0.95 - 0.60 * i / (len(SAM) - 1))
for i, c in enumerate(APEN): COLOR[c] = reds(0.90 - 0.50 * i / (len(APEN) - 1))
COLOR[3126] = "#2e8b57"

D = np.load(NPZ)
lat = D["nlat"].astype(float)
lon = ((D["nlon"].astype(float) + 180) % 360) - 180
parent = D["cnt_340"].astype(float)

S90 = {}
for c in CHILDREN:
    v = D[f"cnt_{c}"].astype(float); o = np.argsort(v)[::-1]; vs = v[o]; pos = vs > 0
    k = int(np.searchsorted(np.cumsum(vs[pos]), 0.90 * vs[pos].sum()) + 1)
    S90[c] = o[:k]
cnt = {}
for c in CHILDREN:
    for n in S90[c]: cnt[n] = cnt.get(n, 0) + 1
shared = [n for n, k in cnt.items() if k > 1]
print(f"90% sets: {sum(len(v) for v in S90.values())} node-memberships, "
      f"{len(cnt)} distinct nodes, shared by >=2 children: {len(shared)}")

fig = plt.figure(figsize=(16, 8.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.06,
                      left=0.03, right=0.99, top=0.91, bottom=0.17)

ax0 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax0.set_global()
ax0.add_feature(cfeature.OCEAN, facecolor=SEA); ax0.add_feature(cfeature.LAND, facecolor=LAND)
ax0.coastlines(linewidth=0.3)
mv = parent > 0
norm = mcolors.LogNorm(vmin=1, vmax=parent.max())
sc = ax0.scatter(lon[mv], lat[mv], c=parent[mv], s=7, cmap="Greys", norm=norm,
                 edgecolor="none", transform=ccrs.PlateCarree(), zorder=3)
ax0.set_title("A. Concept 340's Firing Density", fontweight="bold", fontsize=21)
cax = ax0.inset_axes([0.15, -0.17, 0.7, 0.055])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
ticks = [1, 10, 100, 1000, int(parent.max())]
cb.set_ticks(ticks); cb.set_ticklabels([f"{t:,}" for t in ticks]); cb.minorticks_off()
cb.ax.tick_params(labelsize=15)
cb.set_label("active timesteps (of 8,000)", fontsize=16)

ax1 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
ax1.set_extent([-180, 180, -90, -20], ccrs.PlateCarree())
th = np.linspace(0, 2 * np.pi, 200)
ax1.set_boundary(mpath.Path(np.c_[np.sin(th), np.cos(th)] * 0.5 + 0.5), transform=ax1.transAxes)
ax1.add_feature(cfeature.OCEAN, facecolor=SEA); ax1.add_feature(cfeature.LAND, facecolor=LAND)
ax1.coastlines(linewidth=0.35)
for c in CHILDREN:
    ax1.scatter(lon[S90[c]], lat[S90[c]], s=70, color=COLOR[c], edgecolor="black",
                linewidth=0.5, transform=ccrs.PlateCarree(), zorder=5)
if shared:
    ax1.scatter(lon[shared], lat[shared], s=70, facecolor="none", edgecolor="black",
                linewidth=2.0, transform=ccrs.PlateCarree(), zorder=9)
ax1.set_title("B. Concept 340's Children", fontweight="bold", fontsize=21)
handles = [Line2D([], [], marker="o", ls="none", markersize=12, color=COLOR[c],
                  markeredgecolor="black", label=str(c)) for c in CHILDREN]
ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
           ncol=6, frameon=True, fontsize=14, columnspacing=1.0, handletextpad=0.3)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("WROTE:", OUT)
