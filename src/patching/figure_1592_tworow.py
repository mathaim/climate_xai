"""Two-row 1592 figure: top A ERA5 + B baseline (absolute IVT); bottom C-E dIVT.
Equal-size panels, top row right-flushed, both colorbars aligned on the right.
Env: NPZ, PNG, AMPBETA (default 1.5), DATE. Panel letters forced to A.-E. (bold)."""
import numpy as np, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/patching"
d = np.load(f"{D}/{os.environ.get('NPZ', 'bc_maps_1592.npz')}")
lat, lon = d["lat"], d["lon"]
Tr, B = d["truth"], d["baseline"]
ab = os.environ.get("AMPBETA", "1.5")
DATE = os.environ.get("DATE", "")
if os.environ.get("KEYS"):
    DIF = [d[k] - B for k in os.environ["KEYS"].split(",")]
    dlabels = os.environ.get("DLABELS", "||||").split("||")
else:
    DIF = [d["beta0"] - B, d["beta0.5"] - B, d[f"beta{ab}"] - B]
    dlabels = ["Removed ($\\beta = 0$)", "Clamped ($\\beta = 0.5$)", f"Amplified ($\\beta = {ab}$)"]
def _lab(letter, text):
    body = re.sub(r'^\s*\([a-eA-E]\)\s*', '', text)
    return (f"{letter}. {body}").rstrip()
vabs = float(max(np.nanmax(Tr), np.nanmax(B)))
dmax = max(np.nanpercentile(np.abs(x), 99.8) for x in DIF)
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree(); HAS = True
except Exception:
    proj = None; HAS = False
TITLE, GLBL, CBLBL, CBTK, DTF = 18, 12, 15, 12, 21
fig = plt.figure(figsize=(15, 6.7))
gs = fig.add_gridspec(2, 6, left=0.05, right=0.89, bottom=0.07, top=0.94, wspace=0.08, hspace=0.18)
tops = [fig.add_subplot(gs[0, 1:3], projection=proj), fig.add_subplot(gs[0, 3:5], projection=proj)]
bots = [fig.add_subplot(gs[1, 2*i:2*i+2], projection=proj) for i in range(3)]
def draw(ax, Z, title, cmap, vmn, vmx, left_lab, bot_lab):
    if HAS:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, transform=proj, shading="auto")
        ax.coastlines(resolution="50m", lw=0.7); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = left_lab; gl.bottom_labels = bot_lab
        gl.xlabel_style = {"size": GLBL}; gl.ylabel_style = {"size": GLBL}
    else:
        im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto")
    ax.set_title(title, fontsize=TITLE, loc="left", fontweight="bold"); return im
imA = draw(tops[0], Tr, "A. Ground Truth", "YlGnBu", 0, vabs, True, False)
draw(tops[1], B, "B. Baseline Forecast", "YlGnBu", 0, vabs, False, False)
imD = None
for i, (Z, t) in enumerate(zip(DIF, dlabels)):
    imD = draw(bots[i], Z, _lab("CDE"[i], t), "RdBu_r", -dmax, dmax, i == 0, True)
fig.canvas.draw()
pA = tops[0].get_position(); pT = tops[1].get_position(); pB = bots[2].get_position()
if DATE:
    fig.text((0.05 + pA.x0) / 2.0, (pA.y0 + pA.y1) / 2.0, DATE.replace("\\n", "\n"),
             fontsize=DTF, ha="center", va="center", fontweight="bold")
caxT = fig.add_axes([pB.x1 + 0.012, pT.y0, 0.012, pT.height])
cbT = fig.colorbar(imA, cax=caxT); cbT.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=CBLBL); caxT.tick_params(labelsize=CBTK)
caxB = fig.add_axes([pB.x1 + 0.012, pB.y0, 0.012, pB.height])
cbB = fig.colorbar(imD, cax=caxB); cbB.set_label("$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=CBLBL); caxB.tick_params(labelsize=CBTK)
out = f"/scratch/euh7ys/climate_xai/plots/{os.environ.get('PNG', 'bc_1592_tworow.png')}"
plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.05); print("saved", out)
