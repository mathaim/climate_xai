"""Result A (1829 contains 3481, S-Chile sub-AR) + Result B (99 contains regional AR children) maps.
Encodes matry_L8 over N timesteps, plots per-concept firing locations. Coastlines via cartopy if available."""
import os, glob, numpy as np, torch, datetime as DT, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
N = 400; THRESH = 0.1; OUT = "/scratch/euh7ys/climate_xai/plots"; CC = [1829, 3481, 99, 3392, 1454, 2722]
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature; HAVE = True
except Exception:
    HAVE = False
TK = (lambda: {"transform": ccrs.PlateCarree()} if HAVE else {})
def coast(ax, ext):
    if HAVE:
        ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="0.3"); ax.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="0.6")
        ax.set_extent(ext, crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.grid(alpha=.3)
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(float); nlon = era0[:, lon_i].astype(float)
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy")); rng = np.random.default_rng(0)
    sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    P = {cc: {"lon": [], "lat": [], "act": []} for cc in CC}
    for k, f in enumerate(sel):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for cc in CC:
            fire = acts[:, cc] > THRESH
            if fire.any(): P[cc]["lon"].extend(nlon[fire]); P[cc]["lat"].extend(nlat[fire]); P[cc]["act"].extend(acts[fire, cc])
        if (k + 1) % 100 == 0: print(f"{k+1}/{len(sel)}")
    for cc in CC:
        for kk in P[cc]: P[cc][kk] = np.asarray(P[cc][kk])
    np.savez("/scratch/euh7ys/climate_xai/concept_ivt/footprint_points.npz", **{f"{cc}_{k}": P[cc][k] for cc in CC for k in P[cc]})
    np.savez("/scratch/euh7ys/climate_xai/concept_ivt/footprint_points.npz", **{f"{cc}_{k}": P[cc][k] for cc in CC for k in P[cc]})
    proj = {"projection": ccrs.PlateCarree()} if HAVE else {}
    # Result A: 1829 (broad) contains 3481 (subset), S America
    exA = [-95, -55, -60, -18]; figA = plt.figure(figsize=(6, 7)); axA = figA.add_subplot(111, **proj)
    axA.scatter(P[1829]["lon"], P[1829]["lat"], s=6, c="#2980b9", alpha=0.08, edgecolor="none", label="1829 parent (all firings)", **TK())
    s = P[3481]["act"] > np.quantile(P[3481]["act"], 0.5)
    axA.scatter(P[3481]["lon"][s], P[3481]["lat"][s], s=12, c="#c0392b", edgecolor="none", label="3481 child (strong firings)", **TK())
    coast(axA, exA); axA.set_title("A. 1829 (blue) contains 3481 (red)\nsub-AR coastal moisture, S. Chile ~47-49S"); axA.legend(loc="upper left", fontsize=8)
    figA.savefig(f"{OUT}/nesting_map_final.png", dpi=170, bbox_inches="tight"); print("saved nesting_map_final.png")
    # Result B: 99 global core contains regional AR children
    exB = [-180, 180, -75, 75]; figB = plt.figure(figsize=(12, 6)); axB = figB.add_subplot(111, **proj)
    axB.scatter(P[99]["lon"], P[99]["lat"], s=5, c="0.5", alpha=0.14, edgecolor="none", rasterized=True, label="99 global AR-intensity core", **TK())
    for cc, col, lab in [(3392, "#c0392b", "3392 NW-Pacific AR (IVT 702)"), (1454, "#2980b9", "1454 S-Hemis AR (IVT 522)"), (2722, "#27ae60", "2722 S-Indian AR (IVT 395)")]:
        s = P[cc]["act"] > np.quantile(P[cc]["act"], 0.9)
        axB.scatter(P[cc]["lon"][s], P[cc]["lat"][s], s=9, c=col, edgecolor="none", label=lab, **TK())
    coast(axB, exB); axB.set_title("B. 99 global AR-intensity core (grey) contains regional strong-AR children"); axB.legend(loc="lower left", fontsize=8)
    figB.savefig(f"{OUT}/ar_hierarchy_map.png", dpi=170, bbox_inches="tight"); print("saved ar_hierarchy_map.png")
if __name__ == "__main__":
    main()
