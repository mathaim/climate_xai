"""Baseline + removed/amplified difference panels, with forecast valid time. Drops the
mislabeled 3481 pair."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PATCH = "/scratch/euh7ys/climate_xai/patching"; OUT = "/scratch/euh7ys/climate_xai/plots"
VALID = "valid 22 Aug 2017 00 UTC (init 21 Aug 18 UTC, one 6 h step)"
base = np.load(f"{PATCH}/ivtmap_cap_base.npy"); nlat, nlon = base.shape
lat = np.linspace(-90, 90, nlat); lon = ((np.arange(nlon)*(360.0/nlon) + 180) % 360) - 180
o = np.argsort(lon); lon = lon[o]; prep = lambda a: a[:, o]
try:
    import cartopy.crs as ccrs; proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}; HAVE=True
except Exception: proj = {}; tk = {}; HAVE=False
def trio(cc, amp_tag, ext, fname):
    B = prep(base)
    Dc = prep(np.load(f"{PATCH}/ivtmap_{cc}_clamp.npy")) - B
    Da = prep(np.load(f"{PATCH}/ivtmap_{amp_tag}.npy")) - B
    dmax = max(np.nanpercentile(np.abs(Dc), 99.8), np.nanpercentile(np.abs(Da), 99.8))
    vmax = np.nanpercentile(B, 99.5)
    fig = plt.figure(figsize=(17, 3.9))
    specs = [(B, f"(a) baseline forecast IVT, {VALID}", "YlGnBu", 0, vmax),
             (Dc, f"(b) change when concept {cc} is removed", "RdBu_r", -dmax, dmax),
             (Da, f"(c) change when concept {cc} is amplified (g=3)", "RdBu_r", -dmax, dmax)]
    for i, (Z, title, cmap, vmn, vmx) in enumerate(specs):
        ax = fig.add_subplot(1, 3, i+1, **proj)
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto", **tk)
        if HAVE: ax.coastlines(lw=0.5); ax.set_extent(ext, crs=ccrs.PlateCarree())
        else: ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(title, fontsize=9, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("IVT (kg m$^{-1}$ s$^{-1}$)" if i == 0 else "$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=170, bbox_inches="tight"); print("saved", fname)
trio(99, "99_amp3", [-180, 180, -75, 75], "delta_pair_99.png")
trio(340, "340_amp3", [-95, -55, -60, -18], "delta_pair_340.png")
