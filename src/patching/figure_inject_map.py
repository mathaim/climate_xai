"""Map: injecting SAE concept 1592 into clear-air initial conditions induces an AR. 4 equal panels, one colorbar."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
d = np.load(f"{D}/inject_field_1592.npz"); lat, lon = d["lat"], d["lon"]
vmax = float(np.nanmax(d["inject_b1.0"])); clamp = d["clamp_a1.0"]
specs = [("(a) Baseline (unmodified)",      d["clear"],        0, vmax),
         ("(b) Removed ($g = 0$)",          clamp,             0, vmax),
         ("(c) Written in ($g = 0.6$)",     d["inject_b0.6"],  0, vmax),
         ("(d) Written in ($g = 1.0$)",     d["inject_b1.0"],  0, vmax)]
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree(); HAS = True
except Exception:
    proj = None; HAS = False
fig = plt.figure(figsize=(20, 3.9))
gs = fig.add_gridspec(1, 4, left=0.045, right=0.94, bottom=0.12, top=0.88, wspace=0.02)
axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(4)]
ims = []
for i, (ax, (title, Z, vmn, vmx)) in enumerate(zip(axes, specs)):
    if HAS:
        im = ax.pcolormesh(lon, lat, Z, cmap="YlGnBu", vmin=vmn, vmax=vmx, transform=proj, shading="auto")
        ax.coastlines(resolution="50m", lw=0.7); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        if i > 0: gl.left_labels = False
        gl.xlabel_style = {"size": 15}; gl.ylabel_style = {"size": 15}
    else:
        im = ax.pcolormesh(lon, lat, Z, cmap="YlGnBu", vmin=vmn, vmax=vmx, shading="auto")
        ax.set_xlabel("Longitude (°E)", fontsize=15)
        if i == 0: ax.set_ylabel("Latitude (°N)", fontsize=15)
    ims.append(im); ax.set_title(title, fontsize=19, loc="left")
caxL = fig.add_axes([0.951, 0.14, 0.013, 0.70])
cbL = fig.colorbar(ims[0], cax=caxL); cbL.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=16)
caxL.tick_params(labelsize=14)
out = "/scratch/euh7ys/climate_xai/plots/inject_1592_map.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.06); print("saved", out)
