"""Cross-layer footprints: each layer-8 child (blue) and its best layer-15 match (red).
3481->3160 co-locate (geographic); the AR children 1454/3392/2722 map to L15 latents at
unrelated locations. Encodes L8 and L15 over shared timesteps. No in-image title."""
import os, glob, numpy as np, torch, datetime as DT, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
N = 200; THRESH = 0.1; OUT = "/scratch/euh7ys/climate_xai/plots"; conv = lambda x: x - 360 if x > 180 else x
PAIRS = [(3481, 3160, "3481", "L15 3160  (co-located)"), (1454, 864, "1454", "L15 864"),
         (3392, 3735, "3392", "L15 3735"), (2722, 4036, "2722", "L15 4036")]
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature; HAVE = True
except Exception:
    HAVE = False
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def collect(files, sae, lats, nlon, nlat):
    m, c, fmin, frng = load_sae(sae, "cpu"); pts = {l: {"lon": [], "lat": []} for l in lats}
    for f in files:
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for l in lats:
            fire = acts[:, l] > THRESH
            if fire.any(): pts[l]["lon"].extend(nlon[fire]); pts[l]["lat"].extend(nlat[fire])
    for l in lats: pts[l] = {k: np.asarray(v) for k, v in pts[l].items()}
    return pts
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i]; nlon = np.array([conv(x) for x in era0[:, lon_i]])
    _, c8, _, _ = load_sae("matry_L8", "cpu"); _, c15, _, _ = load_sae("matry_L15", "cpu")
    dtmap = lambda d: {pdt(os.path.basename(f)): f for f in glob.glob(f"{d}/layer*_*.npy")}
    f8, f15 = dtmap(c8["act"]), dtmap(c15["act"]); shared = sorted(set(f8) & set(f15))
    rng = np.random.default_rng(0); sel = [shared[i] for i in rng.choice(len(shared), min(N, len(shared)), replace=False)]
    p8 = collect([f8[d] for d in sel], "matry_L8", [p[0] for p in PAIRS], nlon, nlat)
    p15 = collect([f15[d] for d in sel], "matry_L15", [p[1] for p in PAIRS], nlon, nlat)
    proj = {"projection": ccrs.PlateCarree()} if HAVE else {}; tk = {"transform": ccrs.PlateCarree()} if HAVE else {}
    fig = plt.figure(figsize=(13, 7))
    for i, (l8, l15, n8, n15) in enumerate(PAIRS):
        ax = fig.add_subplot(2, 2, i + 1, **proj)
        ax.scatter(p8[l8]["lon"], p8[l8]["lat"], s=7, c="#2980b9", alpha=0.55, edgecolor="none", label=f"L8 {n8}", **tk)
        ax.scatter(p15[l15]["lon"], p15[l15]["lat"], s=7, c="#c0392b", alpha=0.55, edgecolor="none", label=n15, **tk)
        if HAVE: ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4"); ax.set_global()
        else: ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=.3)
        ax.legend(loc="lower left", fontsize=7, markerscale=1.6)
    fig.tight_layout(); fig.savefig(f"{OUT}/l15_children_footprint.png", dpi=170, bbox_inches="tight"); print("saved l15_children_footprint.png")
if __name__ == "__main__":
    main()
