"""Sufficiency figure: clear day + additive 1592 injection, ABSOLUTE IVT, one shared colorbar.
Env: NPZ (default clear_maps_1592.npz), PNG, DATE, KEYS (3 bottom keys), DLABELS.
Panel letters forced to A.-E. (bold, by panel order) regardless of DLABELS input."""
import numpy as np, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
d = np.load(f"{D}/{os.environ.get('NPZ','clear_maps_1592.npz')}")
lat, lon = d["lat"], d["lon"]; Tr, B = d["truth"], d["baseline"]
DATE = os.environ.get("DATE", "")
keys = os.environ.get("KEYS", "inj1,inj2,inj3").split(",")
PAN = [d[k] for k in keys]
dlabels = os.environ.get("DLABELS", "(c)||(d)||(e)").split("||")
def _lab(letter, text):
    body = re.sub(r'^\s*\([a-eA-E]\)\s*', '', text)   # drop any leading "(x) "
    return (f"{letter}. {body}").rstrip()
vabs = float(max(np.nanmax(Tr), np.nanmax(B), *[np.nanmax(p) for p in PAN]))
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree(); HAS = True
except Exception:
    proj = None; HAS = False
TITLE, GLBL, CBLBL, CBTK, DTF = 18, 12, 15, 12, 21
fig = plt.figure(figsize=(15, 6.7))
gs = fig.add_gridspec(2, 6, left=0.05, right=0.89, bottom=0.07, top=0.94, wspace=0.08, hspace=0.28)
tops = [fig.add_subplot(gs[0, 1:3], projection=proj), fig.add_subplot(gs[0, 3:5], projection=proj)]
bots = [fig.add_subplot(gs[1, 2*i:2*i+2], projection=proj) for i in range(3)]
def draw(ax, Z, title, left_lab, bot_lab):
    if HAS:
        im = ax.pcolormesh(lon, lat, Z, cmap="YlGnBu", vmin=0, vmax=vabs, transform=proj, shading="auto")
        ax.coastlines(resolution="50m", lw=0.7); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False; gl.left_labels = left_lab; gl.bottom_labels = bot_lab
        gl.xlabel_style = {"size": GLBL}; gl.ylabel_style = {"size": GLBL}
    else:
        im = ax.pcolormesh(lon, lat, Z, cmap="YlGnBu", vmin=0, vmax=vabs, shading="auto")
    ax.set_title(title, fontsize=TITLE, loc="left", fontweight="bold"); return im
im = draw(tops[0], Tr, "A. Ground Truth", True, True)
draw(tops[1], B, "B. Baseline Forecast", False, True)
for i, (Z, t) in enumerate(zip(PAN, dlabels)):
    im = draw(bots[i], Z, _lab("CDE"[i], t), i == 0, True)
fig.canvas.draw()
pA = tops[0].get_position(); pT = tops[1].get_position(); pB = bots[2].get_position()
if DATE:
    fig.text((0.05 + pA.x0) / 2.0, (pA.y0 + pA.y1) / 2.0, DATE.replace("\\n", "\n"),
             fontsize=DTF, ha="center", va="center", fontweight="bold")
cax = fig.add_axes([pB.x1 + 0.012, pB.y0, 0.013, pT.y1 - pB.y0])
cb = fig.colorbar(im, cax=cax); cb.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=CBLBL); cax.tick_params(labelsize=CBTK)
out = f"/scratch/euh7ys/climate_xai/plots/{os.environ.get('PNG','clear_1592_suff.png')}"
plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.05); print("saved", out)
