"""Publication map: injecting SAE concept 1592 into clear-air initial conditions induces an AR."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
d = np.load(f"{D}/inject_field_1592.npz"); lat, lon = d["lat"], d["lon"]
vmax = float(np.nanmax(d["inject_b1.0"]))
panels = [("(a) Baseline forecast \u2014 no injection", d["clear"], "YlGnBu", 0, vmax),
          ("(b) Inject concept 1592  (b = 0.6, real-AR level)", d["inject_b0.6"], "YlGnBu", 0, vmax),
          ("(c) Inject concept 1592  (b = 1.0)", d["inject_b1.0"], "YlGnBu", 0, vmax),
          ("(d) Moisture transport added:  (c) \u2212 (a)", d["inject_b1.0"] - d["clear"], "YlOrRd", 0, None)]
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree(); HAS = True
except Exception:
    HAS = False
fig, axes = plt.subplots(1, 4, figsize=(19, 6), subplot_kw=dict(projection=proj) if HAS else None)
for i, (ax, (title, Z, cmap, vmn, vmx)) in enumerate(zip(axes, panels)):
    vmx = vmx if vmx is not None else float(np.nanmax(Z))
    if HAS:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, transform=proj, shading="auto")
        ax.coastlines(resolution="50m", lw=0.7); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.plot(-123.1, 49.3, marker="*", color="red", ms=15, transform=proj, zorder=6)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        if i > 0: gl.left_labels = False
        gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
    else:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto")
        ax.plot(-123.1, 49.3, marker="*", color="red", ms=15)
        ax.set_xlabel("Longitude (\u00b0E)", fontsize=9)
        if i == 0: ax.set_ylabel("Latitude (\u00b0N)", fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left")
    if i == 2: fig.colorbar(im, ax=ax, shrink=0.62, pad=0.03, label="IVT (kg m$^{-1}$ s$^{-1}$)")
    if i == 3: fig.colorbar(im, ax=ax, shrink=0.62, pad=0.03, label="$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)")
fig.suptitle("Injecting a single SAE concept into clear-air initial conditions induces a landfalling atmospheric river in GraphCast",
             fontsize=13, y=0.97)
cap = ("GraphCast 6 h forecast of vertically integrated vapour transport (IVT) over the northeast Pacific from a calm "
       "summer initial state (15 Jul 2021 00 UTC). (a) Unmodified forecast. (b, c) Forecast after additively injecting "
       "Plain-L8 SAE concept 1592 into the mesh activations at processor step 8 \u2014 at the level the concept reaches "
       "during real atmospheric rivers (b) and at double that (c). (d) Difference (c) minus (a), isolating the moisture "
       "transport produced by the injection. Panels (a)\u2013(c) share the colour scale. Red star: Vancouver, British Columbia.")
fig.text(0.5, 0.04, cap, ha="center", va="top", fontsize=8.8, wrap=True, color="#333")
fig.subplots_adjust(bottom=0.22, top=0.88)
out = "/scratch/euh7ys/climate_xai/plots/inject_1592_map.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
