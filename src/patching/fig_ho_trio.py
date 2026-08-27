"""Held-out (Nov 2021) trio for concept 99: baseline / removed / amplified. Shared dIVT colorbar for (b),(c)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PATCH = "/scratch/euh7ys/climate_xai/patching"; OUT = "/scratch/euh7ys/climate_xai/plots"
base = np.load(f"{PATCH}/ivtmap_ho_base.npy"); nlat, nlon = base.shape
lat = np.linspace(-90, 90, nlat); lon = ((np.arange(nlon)*(360.0/nlon) + 180) % 360) - 180
o = np.argsort(lon); lon = lon[o]; prep = lambda a: a[:, o]
try:
    import cartopy.crs as ccrs; proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}; HAVE=True
except Exception: proj = {}; tk = {}; HAVE=False
B = prep(base)
Dc = prep(np.load(f"{PATCH}/ivtmap_ho_99_clamp.npy")) - B
Da = prep(np.load(f"{PATCH}/ivtmap_ho_99_amp3.npy")) - B
dmax = max(np.nanpercentile(np.abs(Dc), 99.8), np.nanpercentile(np.abs(Da), 99.8))
specs = [(B,  "(a) baseline IVT ($g=1$)",              "YlGnBu", 0, np.nanpercentile(B, 99.5)),
         (Dc, "(b) concept 99 removed ($g=0$)",  "RdBu_r", -dmax, dmax),
         (Da, "(c) concept 99 amplified ($g=3$)","RdBu_r", -dmax, dmax)]
fig, axes = plt.subplots(1, 3, figsize=(19, 4.2), subplot_kw=proj, constrained_layout=True)
ims = []
for ax, (Z, title, cmap, vmn, vmx) in zip(axes, specs):
    im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto", **tk); ims.append(im)
    if HAVE: ax.coastlines(lw=0.5); ax.set_global()
    ax.set_title(title, fontsize=17, loc="left")
cb0 = fig.colorbar(ims[0], ax=axes[0], location="left", shrink=0.86, pad=0.012)
cb0.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=15); cb0.ax.tick_params(labelsize=13)
cbd = fig.colorbar(ims[2], ax=[axes[1], axes[2]], location="right", shrink=0.86, pad=0.012)
cbd.set_label("$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=15); cbd.ax.tick_params(labelsize=13)
fig.savefig(f"{OUT}/delta_pair_99_heldout.png", dpi=170, bbox_inches="tight")
print("saved delta_pair_99_heldout.png")
