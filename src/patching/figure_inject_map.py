"""Map: injecting SAE concept 1592 into clear-air initial conditions induces an AR. Equal panels, edge colorbars."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
d = np.load(f"{D}/inject_field_1592.npz"); lat, lon = d["lat"], d["lon"]
vmax = float(np.nanmax(d["inject_b1.0"])); diff = d["inject_b1.0"] - d["clear"]
specs = [("(a) Baseline forecast \u2014 no injection", d["clear"], "YlGnBu", 0, vmax),
         ("(b) Inject concept 1592  (b = 0.6, real-AR level)", d["inject_b0.6"], "YlGnBu", 0, vmax),
         ("(c) Inject concept 1592  (b = 1.0)", d["inject_b1.0"], "YlGnBu", 0, vmax),
         ("(d) Moisture added:  (c) \u2212 (a)", diff, "YlOrRd", 0, float(np.nanmax(diff)))]
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree(); HAS = True
except Exception:
    proj = None; HAS = False
fig = plt.figure(figsize=(18, 5))
gs = fig.add_gridspec(1, 4, left=0.10, right=0.92, bottom=0.10, top=0.93, wspace=0.06)
axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(4)]
ims = []
for i, (ax, (title, Z, cmap, vmn, vmx)) in enumerate(zip(axes, specs)):
    if HAS:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, transform=proj, shading="auto")
        ax.coastlines(resolution="50m", lw=0.7); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.plot(-123.1, 49.3, marker="*", color="red", ms=14, transform=proj, zorder=6)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        if i > 0: gl.left_labels = False
        gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
    else:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto")
        ax.plot(-123.1, 49.3, marker="*", color="red", ms=14); ax.set_xlabel("Longitude (\u00b0E)", fontsize=8)
        if i == 0: ax.set_ylabel("Latitude (\u00b0N)", fontsize=8)
    ims.append(im); ax.set_title(title, fontsize=10.5, loc="left")
# left colorbar: shared IVT (a-c)
caxL = fig.add_axes([0.035, 0.28, 0.012, 0.45])
cbL = fig.colorbar(ims[0], cax=caxL); cbL.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=9)
caxL.yaxis.set_ticks_position("left"); caxL.yaxis.set_label_position("left")
# right colorbar: difference (d)
caxR = fig.add_axes([0.945, 0.28, 0.012, 0.45])
cbR = fig.colorbar(ims[3], cax=caxR); cbR.set_label("$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=9)
out = "/scratch/euh7ys/climate_xai/plots/inject_1592_map.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180); print("saved", out)
