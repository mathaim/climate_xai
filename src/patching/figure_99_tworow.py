"""Two-row concept-99 GLOBAL figure: top A ERA5 + B baseline (absolute IVT); bottom C-E dIVT.
Equal-size panels, top right-flushed, both colorbars aligned on the right. Env: PNG, DATE.
Panel letters A.-E. (no parens); full title bold."""
import numpy as np, os, re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PATCH = "/scratch/euh7ys/climate_xai/patching"; OUT = "/scratch/euh7ys/climate_xai/plots"
base = np.load(f"{PATCH}/ivtmap_ho_base.npy"); nlat, nlon = base.shape
lat = np.linspace(-90, 90, nlat); lon = ((np.arange(nlon)*(360.0/nlon) + 180) % 360) - 180
o = np.argsort(lon); lon = lon[o]; prep = lambda a: a[:, o]
Tr = prep(np.load(f"{PATCH}/ivtmap_ho_truth.npy")); B = prep(base)
DIF = [prep(np.load(f"{PATCH}/ivtmap_ho_99_clamp.npy")) - B,
       prep(np.load(f"{PATCH}/ivtmap_ho_99_beta05.npy")) - B,
       prep(np.load(f"{PATCH}/ivtmap_ho_99_amp3.npy")) - B]
dlabels = ["Removed ($\\beta = 0$)", "Clamped ($\\beta = 0.5$)", "Amplified ($\\beta = 3$)"]
DATE = os.environ.get("DATE", "")
def _lab(letter, text):
    body = re.sub(r'^\s*\([a-eA-E]\)\s*', '', text)
    return (f"{letter}. {body}").rstrip()
vabs = float(np.nanpercentile(B, 99.5))
dmax = max(np.nanpercentile(np.abs(x), 99.8) for x in DIF)
try:
    import cartopy.crs as ccrs
    proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}; HAVE = True
except Exception:
    proj = {}; tk = {}; HAVE = False
TITLE, CBLBL, CBTK, DTF = 18, 15, 12, 21
fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(2, 6, left=0.05, right=0.89, bottom=0.04, top=0.95, wspace=0.06, hspace=0.12)
tops = [fig.add_subplot(gs[0, 1:3], **proj), fig.add_subplot(gs[0, 3:5], **proj)]
bots = [fig.add_subplot(gs[1, 2*i:2*i+2], **proj) for i in range(3)]
def draw(ax, Z, title, cmap, vmn, vmx):
    im = ax.pcolormesh(lon, lat, Z, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto", **tk)
    if HAVE: ax.coastlines(lw=0.5); ax.set_global()
    ax.set_title(title, fontsize=TITLE, loc="left", fontweight="bold"); return im
imA = draw(tops[0], Tr, "A. Ground Truth", "YlGnBu", 0, vabs)
draw(tops[1], B, "B. Baseline Forecast", "YlGnBu", 0, vabs)
imD = None
for i, (Z, t) in enumerate(zip(DIF, dlabels)):
    imD = draw(bots[i], Z, _lab("CDE"[i], t), "RdBu_r", -dmax, dmax)
fig.canvas.draw()
pA = tops[0].get_position(); pT = tops[1].get_position(); pB = bots[2].get_position()
if DATE:
    fig.text((0.05 + pA.x0) / 2.0, (pA.y0 + pA.y1) / 2.0, DATE.replace("\\n", "\n"),
             fontsize=DTF, ha="center", va="center", fontweight="bold")
caxT = fig.add_axes([pB.x1 + 0.012, pT.y0, 0.012, pT.height])
cbT = fig.colorbar(imA, cax=caxT); cbT.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=CBLBL); caxT.tick_params(labelsize=CBTK)
caxB = fig.add_axes([pB.x1 + 0.012, pB.y0, 0.012, pB.height])
cbB = fig.colorbar(imD, cax=caxB); cbB.set_label("$\\Delta$IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=CBLBL); caxB.tick_params(labelsize=CBTK)
out = f"{OUT}/{os.environ.get('PNG', 'tworow_99_heldout.png')}"
fig.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out)
