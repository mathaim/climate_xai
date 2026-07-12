"""Held-out (Nov 2021) trio for concept 99: baseline / removed / amplified."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PATCH = "/scratch/euh7ys/climate_xai/patching"; OUT = "/scratch/euh7ys/climate_xai/plots"
VALID = "valid 15 Nov 2021 18 UTC (init 15 Nov 12 UTC); date held out from SAE training"
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
fig = plt.figure(figsize=(17, 3.9))
for i, (Z, title, cmap, vmn, vmx) in enumerate([
        (B, "(a) baseline forecast IVT ($g=1$)", "YlGnBu", 0, np.nanpercentile(B, 99.5)),
        (Dc, "(b) change when concept 99 is removed ($g=0$)", "RdBu_r", -dmax, dmax),
        (Da, "(c) change when concept 99 is amplified (g=3)", "RdBu_r", -dmax, dmax)]):
    ax = fig.add_subplot(1, 3, i+1, **proj)
    im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto", **tk)
    if HAVE: ax.coastlines(lw=0.5); ax.set_global()
    ax.set_title(title, fontsize=8.5, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("IVT (kg m$^{-1}$ s$^{-1}$)" if i == 0 else "$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/delta_pair_99_heldout.png", dpi=170, bbox_inches="tight")
print("saved delta_pair_99_heldout.png")
