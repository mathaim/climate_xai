"""Map: injecting SAE concept 1592 into clear air conjures an atmospheric river over W. North America."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
d = np.load(f"{D}/inject_field_1592.npz"); lat, lon = d["lat"], d["lon"]
vmax = float(np.nanmax(d["inject_b1.0"]))
panels = [("clear", "clear-air baseline\n(calm July day)", d["clear"], "YlGnBu", 0, vmax),
          ("inject_b0.6", "inject 1592, realistic level\n(b=0.6)", d["inject_b0.6"], "YlGnBu", 0, vmax),
          ("inject_b1.0", "inject 1592, strong\n(b=1.0)", d["inject_b1.0"], "YlGnBu", 0, vmax),
          ("diff", "conjured moisture\n(strong minus clear)", d["inject_b1.0"] - d["clear"], "YlOrRd", 0, None)]
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree(); HAS = True
except Exception:
    HAS = False
fig, axes = plt.subplots(1, 4, figsize=(19, 5), subplot_kw=dict(projection=proj) if HAS else None)
for ax, (k, title, Z, cmap, vmn, vmx) in zip(axes, panels):
    vmx = vmx if vmx is not None else float(np.nanmax(Z))
    if HAS:
        m = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, transform=proj, shading="auto")
        ax.coastlines(resolution="50m", lw=0.7); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.plot(-123.1, 49.3, marker="*", color="red", ms=15, transform=proj)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
    else:
        m = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto")
        ax.plot(-123.1, 49.3, marker="*", color="red", ms=15); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"{title}   (max {np.nanmax(Z):.0f})", fontsize=9.5)
    fig.colorbar(m, ax=ax, shrink=0.55, pad=0.03)
fig.suptitle("Injecting one SAE concept (1592) into a clear-air day conjures an atmospheric river aimed at British Columbia (red star)", fontsize=12.5)
out = "/scratch/euh7ys/climate_xai/plots/inject_1592_map.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
