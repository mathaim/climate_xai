"""Output-map triptychs from run_capstone's saved IVT fields: baseline / amplified / difference.
99: global. 340: South-America zoom (drying, diverging palette)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PATCH = "/scratch/euh7ys/climate_xai/patching"; OUT = "/scratch/euh7ys/climate_xai/plots"
base = np.load(f"{PATCH}/ivtmap_cap_base.npy")
nlat, nlon = base.shape
lat = np.linspace(-90, 90, nlat)
lon0 = np.arange(nlon) * (360.0 / nlon); lon = ((lon0 + 180) % 360) - 180
order = np.argsort(lon); lon = lon[order]
def prep(a): return a[:, order]
try:
    import cartopy.crs as ccrs; proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}; HAVE=True
except Exception: proj = {}; tk = {}; HAVE=False
def triptych(tag, ext, fname, dcmap, dsym):
    amp = np.load(f"{PATCH}/ivtmap_{tag}.npy")
    B, A = prep(base), prep(amp); D = A - B
    vmax = np.nanpercentile(B, 99.5)
    dmax = np.nanpercentile(np.abs(D), 99.5)
    fig = plt.figure(figsize=(16, 3.6))
    for i, (Z, title, cmap, vmn, vmx) in enumerate([
            (B, "(a) baseline forecast", "YlGnBu", 0, vmax),
            (A, "(b) concept amplified (g=3)", "YlGnBu", 0, vmax),
            (D, "(c) change: (b) minus (a)", dcmap, -dmax if dsym else 0, dmax)]):
        ax = fig.add_subplot(1, 3, i+1, **proj)
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto", **tk)
        if HAVE: ax.coastlines(lw=0.5); ax.set_extent(ext, crs=ccrs.PlateCarree())
        else: ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(title, fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=170, bbox_inches="tight"); print("saved", fname)
triptych("99_amp3", [-180, 180, -75, 75], "output_map_99.png", "RdBu_r", True)
triptych("340_amp3", [-95, -55, -60, -18], "output_map_340.png", "RdBu_r", True)
