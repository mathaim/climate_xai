"""Five panels for 1592: truth + baseline absolute; interventions absolute or as dIVT.
Env: NPZ (input), PNG (output), MODE=absolute|delta."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MODE = os.environ.get("MODE", "absolute")
D = "/scratch/euh7ys/climate_xai/patching"
d = np.load(f"{D}/{os.environ.get('NPZ', 'bc_maps_1592.npz')}")
lat, lon = d["lat"], d["lon"]
Tr, B = d["truth"], d["baseline"]
I = [d["beta0"], d["beta0.5"], d["beta1.5"]]
labels = ["(c) Removed ($\\beta = 0$)", "(d) Clamped ($\\beta = 0.5$)", "(e) Amplified ($\\beta = 1.5$)"]
vabs = float(max(np.nanmax(Tr), np.nanmax(B)))
if MODE == "delta":
    DIF = [x - B for x in I]
    dmax = max(np.nanpercentile(np.abs(x), 99.8) for x in DIF)
    tail = [(Z, t, "RdBu_r", -dmax, dmax) for Z, t in zip(DIF, labels)]
else:
    tail = [(Z, t, "YlGnBu", 0, vabs) for Z, t in zip(I, labels)]
specs = [(Tr, "(a) ERA5 (observed)",   "YlGnBu", 0, vabs),
         (B,  "(b) Baseline forecast", "YlGnBu", 0, vabs)] + tail
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree(); HAS = True
except Exception:
    proj = None; HAS = False
fig = plt.figure(figsize=(25, 3.9))
gs = fig.add_gridspec(1, 5, left=0.055, right=0.93, bottom=0.12, top=0.88, wspace=0.02)
axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(5)]
ims = []
for i, (Z, title, cmap, vmn, vmx) in enumerate(specs):
    ax = axes[i]
    if HAS:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, transform=proj, shading="auto")
        ax.coastlines(resolution="50m", lw=0.7); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        if i > 0: gl.left_labels = False
        gl.xlabel_style = {"size": 15}; gl.ylabel_style = {"size": 15}
    else:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto")
    ims.append(im); ax.set_title(title, fontsize=18, loc="left")
caxL = fig.add_axes([0.012, 0.14, 0.011, 0.70])
cbL = fig.colorbar(ims[0], cax=caxL); cbL.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=15)
caxL.tick_params(labelsize=13); caxL.yaxis.set_ticks_position("left"); caxL.yaxis.set_label_position("left")
if MODE == "delta":
    caxR = fig.add_axes([0.94, 0.14, 0.011, 0.70])
    cbR = fig.colorbar(ims[4], cax=caxR)
    cbR.set_label("$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=15); caxR.tick_params(labelsize=13)
out = f"/scratch/euh7ys/climate_xai/plots/{os.environ.get('PNG', 'bc_1592_fivepanel.png')}"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.06)
print("saved", out, "| MODE =", MODE)
