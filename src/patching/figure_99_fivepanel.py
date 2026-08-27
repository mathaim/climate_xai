"""Five panels for concept 99: ERA5 truth, baseline, removed(b0), clamped(b0.5), amplified(b3).
MODE='delta' shows the three interventions as dIVT (recommended); 'absolute' shows raw IVT."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
MODE = "delta"
PATCH = "/scratch/euh7ys/climate_xai/patching"; OUT = "/scratch/euh7ys/climate_xai/plots"
base = np.load(f"{PATCH}/ivtmap_ho_base.npy"); nlat, nlon = base.shape
lat = np.linspace(-90, 90, nlat); lon = ((np.arange(nlon)*(360.0/nlon) + 180) % 360) - 180
o = np.argsort(lon); lon = lon[o]; prep = lambda a: a[:, o]
try:
    import cartopy.crs as ccrs; proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}; HAVE=True
except Exception: proj = {}; tk = {}; HAVE=False
Tr = prep(np.load(f"{PATCH}/ivtmap_ho_truth.npy")); B = prep(base)
C0  = prep(np.load(f"{PATCH}/ivtmap_ho_99_clamp.npy"))
C05 = prep(np.load(f"{PATCH}/ivtmap_ho_99_beta05.npy"))
A3  = prep(np.load(f"{PATCH}/ivtmap_ho_99_amp3.npy"))
vabs = np.nanpercentile(B, 99.5)
if MODE == "delta":
    D = [C0-B, C05-B, A3-B]
    dmax = max(np.nanpercentile(np.abs(x), 99.8) for x in D)
    tail = [( D[0], "(c) removed ($\\beta=0$)",    "RdBu_r", -dmax, dmax),
            ( D[1], "(d) clamped ($\\beta=0.5$)",  "RdBu_r", -dmax, dmax),
            ( D[2], "(e) amplified ($\\beta=3$)",  "RdBu_r", -dmax, dmax)]
else:
    tail = [( C0,  "(c) removed ($\\beta=0$)",    "YlGnBu", 0, vabs),
            ( C05, "(d) clamped ($\\beta=0.5$)",  "YlGnBu", 0, vabs),
            ( A3,  "(e) amplified ($\\beta=3$)",  "YlGnBu", 0, vabs)]
specs = [(Tr, "(a) ERA5 (observed)",    "YlGnBu", 0, vabs),
         (B,  "(b) baseline forecast",  "YlGnBu", 0, vabs)] + tail
fig, axes = plt.subplots(1, 5, figsize=(30, 4.2), subplot_kw=proj, constrained_layout=True)
ims = []
for ax, (Z, title, cmap, vmn, vmx) in zip(axes, specs):
    im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto", **tk); ims.append(im)
    if HAVE: ax.coastlines(lw=0.5); ax.set_global()
    ax.set_title(title, fontsize=17, loc="left")
cb0 = fig.colorbar(ims[0], ax=axes[0], location="left", shrink=0.86, pad=0.012)
cb0.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=15); cb0.ax.tick_params(labelsize=13)
if MODE == "delta":
    cbd = fig.colorbar(ims[4], ax=list(axes[2:]), location="right", shrink=0.86, pad=0.012)
    cbd.set_label("$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=15); cbd.ax.tick_params(labelsize=13)
suffix = "" if MODE == "delta" else "_abs"
fig.savefig(f"{OUT}/fivepanel_99_heldout{suffix}.png", dpi=170, bbox_inches="tight")
print(f"saved fivepanel_99_heldout{suffix}.png")
