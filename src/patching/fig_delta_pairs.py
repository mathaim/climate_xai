"""Difference-pair figures: dIVT(removed) vs dIVT(amplified), shared diverging scale."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PATCH = "/scratch/euh7ys/climate_xai/patching"; OUT = "/scratch/euh7ys/climate_xai/plots"
base = np.load(f"{PATCH}/ivtmap_cap_base.npy"); nlat, nlon = base.shape
lat = np.linspace(-90, 90, nlat); lon = ((np.arange(nlon)*(360.0/nlon) + 180) % 360) - 180
o = np.argsort(lon); lon = lon[o]; prep = lambda a: a[:, o]
try:
    import cartopy.crs as ccrs; proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}; HAVE=True
except Exception: proj = {}; tk = {}; HAVE=False
def pair(cc, amp_tag, ext, fname):
    Dc = prep(np.load(f"{PATCH}/ivtmap_{cc}_clamp.npy")) - prep(base)
    Da = prep(np.load(f"{PATCH}/ivtmap_{amp_tag}.npy")) - prep(base)
    dmax = max(np.nanpercentile(np.abs(Dc), 99.8), np.nanpercentile(np.abs(Da), 99.8))
    fig = plt.figure(figsize=(13, 3.8))
    for i, (Z, title) in enumerate([(Dc, f"(a) concept {cc} removed"), (Da, f"(b) concept {cc} amplified (g=3)")]):
        ax = fig.add_subplot(1, 2, i+1, **proj)
        im = ax.pcolormesh(lon, lat, Z, cmap="RdBu_r", vmin=-dmax, vmax=dmax, shading="auto", **tk)
        if HAVE: ax.coastlines(lw=0.5); ax.set_extent(ext, crs=ccrs.PlateCarree())
        else: ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(title, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=fig.axes, shrink=0.85, pad=0.02); cb.set_label("$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=9)
    fig.savefig(f"{OUT}/{fname}", dpi=170, bbox_inches="tight"); print("saved", fname)
pair(99, "99_amp3", [-180, 180, -75, 75], "delta_pair_99.png")
pair(340, "340_amp3", [-95, -55, -60, -18], "delta_pair_340.png")
pair(3481, "3481_clamp", [-95, -55, -60, -18], "delta_pair_3481.png")   # clamp-only for child: both panels same file OK to ignore
