"""All 4 AR regions on a global map; also prints each region's extent and approx area."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.analysis.ar_intensity.regions import REGIONS
PLOTS = "/scratch/euh7ys/climate_xai/plots"
conv = lambda x: x - 360 if x > 180 else x
COLORS = {"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
NAME = {"W_N_America":"Western North America","W_Europe":"Western Europe",
        "W_S_America":"Western South America","E_Australia":"Eastern Australia"}
def main():
    fig = plt.figure(figsize=(15, 8)); ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global(); ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
    ax.add_feature(cfeature.OCEAN, facecolor="#e3edf3")
    ax.coastlines("110m", color="#666", lw=.5); ax.gridlines(lw=.3)
    print(f"{'region':<22}{'lat span':>11}{'lon width':>11}{'~area Mkm2':>12}")
    for r, cfg in REGIONS.items():
        la = cfg["lat"]; lons = cfg["lon"]
        latspan = la[1] - la[0]; lonw = sum(x1 - x0 for x0, x1 in lons); ml = (la[0] + la[1]) / 2
        area = latspan * lonw * 111 * 111 * np.cos(np.radians(ml)) / 1e6
        print(f"{r:<22}{latspan:>8.1f}deg{lonw:>8.1f}deg{area:>10.2f}")
        xx = [conv(x) for seg in lons for x in seg]; bx0, bx1 = min(xx), max(xx)
        ax.plot([bx0, bx1, bx1, bx0, bx0], [la[0], la[0], la[1], la[1], la[0]], c=COLORS[r], lw=2.4,
                transform=ccrs.PlateCarree(), zorder=6)
        ax.text(conv(lons[0][0]), la[1] + 2, NAME[r], color=COLORS[r], fontsize=10,
                fontweight="bold", transform=ccrs.PlateCarree(), zorder=7)
    fig.savefig(f"{PLOTS}/regions_global_map.png", dpi=160, bbox_inches="tight")
    print("saved", f"{PLOTS}/regions_global_map.png")
if __name__ == "__main__":
    main()
