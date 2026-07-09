"""Three-panel AR hierarchy across depth: each layer's AR core (grey, all firings) + its
contained AR children (colored, strong firings). L0/L15 children selected programmatically
(P(core|c)>=0.5, corr>=0.30, cnt>=500, index>=1024); L8 uses the case-study children.
NMAX stratified timesteps (default 5000), GPU-aware."""
import os, glob, numpy as np, torch, datetime as DT, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
N = int(os.environ.get("NMAX", "5000")); THRESH = 0.1
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("REQUIRE_GPU") == "1": assert torch.cuda.is_available()
OUT = "/scratch/euh7ys/climate_xai/plots"
C = "/scratch/euh7ys/climate_xai/concept_ivt"; conv = lambda x: x - 360 if x > 180 else x
COLS = ["#c0392b", "#2980b9", "#27ae60", "#e67e22"]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")
def pick_children(scan_npz, corr_npy, core):
    d = np.load(scan_npz); both, cnt = d["both"], d["cnt"]
    corr = np.load(corr_npy); P = both[core] / np.maximum(cnt, 1)
    cand = [c for c in range(1024, 4096) if cnt[c] >= 500 and corr[c] >= 0.30 and P[c] >= 0.5]
    cand = sorted(cand, key=lambda c: -corr[c])[:3]
    return cand, {c: (round(float(corr[c]),2), round(float(P[c]),2)) for c in cand}
def footprints(sae, CC):
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(float); nlon = np.array([conv(x) for x in era0[:, lon_i].astype(float)])
    m, c, fmin, frng = load_sae(sae, DEV)
    files = sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
    sel = files if N >= len(files) else [files[i] for i in np.linspace(0, len(files)-1, N).astype(int)]
    P = {cc: {"lon": [], "lat": [], "act": []} for cc in CC}
    for k, f in enumerate(sel):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(DEV)).cpu().numpy()
        for cc in CC:
            fire = acts[:, cc] > THRESH
            if fire.any(): P[cc]["lon"].extend(nlon[fire]); P[cc]["lat"].extend(nlat[fire]); P[cc]["act"].extend(acts[fire, cc])
        if (k+1) % 1000 == 0: print(f"  {sae} {k+1}/{len(sel)}", flush=True)
    return {cc: {k2: np.asarray(v) for k2, v in P[cc].items()} for cc in CC}
def main():
    print("device", DEV, "N", N, flush=True)
    ch0, m0 = pick_children(f"{C}/nesting_scan_matry_L0.npz", f"{C}/ar_corr_matry_L0.npy", 139)
    ch15, m15 = pick_children(f"{C}/nesting_scan_matry_L15.npz", f"{C}/ar_corr_matry_L15.npy", 111)
    print("L0 139 children:", m0, flush=True); print("L15 111 children:", m15, flush=True)
    PANELS = [("matry_L0", 139, ch0, "layer 0"), ("matry_L8", 99, [1454, 3392, 2722], "layer 8"),
              ("matry_L15", 111, ch15, "layer 15")]
    try:
        import cartopy.crs as ccrs, cartopy.feature as cfeature; HAVE = True
        proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}
    except Exception: HAVE = False; proj = {}; tk = {}
    fig = plt.figure(figsize=(11, 13.5)); cache = {}
    for i, (sae, core, chs, lab) in enumerate(PANELS):
        pts = footprints(sae, [core] + chs)
        for cc in [core] + chs:
            for k2 in ("lon","lat","act"): cache[f"{sae}_{cc}_{k2}"] = pts[cc][k2]
        ax = fig.add_subplot(3, 1, i + 1, **proj)
        ax.scatter(pts[core]["lon"], pts[core]["lat"], s=4, c="0.5", alpha=0.12, edgecolor="none", rasterized=True,
                   label=f"{core}  AR core", **tk)
        for cc, col in zip(chs, COLS):
            if len(pts[cc]["act"]) == 0: continue
            s = pts[cc]["act"] >= np.quantile(pts[cc]["act"], 0.9)
            ax.scatter(pts[cc]["lon"][s], pts[cc]["lat"][s], s=8, c=col, edgecolor="none", label=f"{cc}  child", **tk)
        if HAVE: ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.35"); ax.set_global()
        else: ax.set_xlim(-180, 180); ax.set_ylim(-85, 85); ax.grid(alpha=.3)
        ax.set_title(lab, fontsize=11); ax.legend(loc="lower left", fontsize=7.5, markerscale=1.4)
    np.savez(f"{C}/ar_hierarchy_depth_points.npz", **cache)
    fig.tight_layout(); fig.savefig(f"{OUT}/ar_hierarchy_depth.png", dpi=170, bbox_inches="tight")
    print("saved ar_hierarchy_depth.png", flush=True)
if __name__ == "__main__":
    main()
