"""L15 analogue of ar_hierarchy_map: core 111 (grey, all firings) + its localized AR children
(colored, strong firings). Children selected by the same criteria as 99's: outer group, AR corr
high, contained in 111 (from nesting_scan_matry_L15.npz). Then encodes L15 footprints."""
import os, glob, numpy as np, torch, datetime as DT, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
N = 400; THRESH = 0.1; OUT = "/scratch/euh7ys/climate_xai/plots"
C = "/scratch/euh7ys/climate_xai/concept_ivt"; conv = lambda x: x - 360 if x > 180 else x
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")
def main():
    # --- select 111's children: outer, AR-correlated, contained in 111 ---
    d = np.load(f"{C}/nesting_scan_matry_L15.npz"); both, cnt = d["both"], d["cnt"]
    corr = np.load(f"{C}/ar_corr_matry_L15.npy")
    P111 = both[111] / np.maximum(cnt, 1)
    cand = [c for c in range(1024, 4096) if cnt[c] >= 500 and corr[c] >= 0.30 and P111[c] >= 0.5]
    cand = sorted(cand, key=lambda c: -corr[c])[:4]
    print("children of 111 selected:", [(c, round(float(corr[c]),2), round(float(P111[c]),2), int(cnt[c])) for c in cand], flush=True)
    CC = [111] + cand
    # --- encode footprints ---
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(float); nlon = np.array([conv(x) for x in era0[:, lon_i].astype(float)])
    m, c, fmin, frng = load_sae("matry_L15", "cpu")
    files = sorted(glob.glob(f"{c['act']}/layer*_*.npy")); rng = np.random.default_rng(0)
    sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    Pts = {cc: {"lon": [], "lat": [], "act": []} for cc in CC}
    for k, f in enumerate(sel):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for cc in CC:
            fire = acts[:, cc] > THRESH
            if fire.any(): Pts[cc]["lon"].extend(nlon[fire]); Pts[cc]["lat"].extend(nlat[fire]); Pts[cc]["act"].extend(acts[fire, cc])
        if (k+1) % 100 == 0: print(f"  {k+1}/{len(sel)}", flush=True)
    for cc in CC: Pts[cc] = {k2: np.asarray(v) for k2, v in Pts[cc].items()}
    try:
        import cartopy.crs as ccrs, cartopy.feature as cfeature; HAVE = True
        proj = {"projection": ccrs.PlateCarree()}; tk = {"transform": ccrs.PlateCarree()}
    except Exception: HAVE = False; proj = {}; tk = {}
    fig = plt.figure(figsize=(12, 6)); ax = fig.add_subplot(111, **proj)
    ax.scatter(Pts[111]["lon"], Pts[111]["lat"], s=5, c="0.5", alpha=0.14, edgecolor="none", rasterized=True,
               label="111  L15 AR-intensity core", **tk)
    cols = ["#c0392b", "#2980b9", "#27ae60", "#e67e22"]
    for cc, col in zip(CC[1:], cols):
        s = Pts[cc]["act"] >= np.quantile(Pts[cc]["act"], 0.9)
        ax.scatter(Pts[cc]["lon"][s], Pts[cc]["lat"][s], s=9, c=col, edgecolor="none", label=f"{cc}  child", **tk)
    if HAVE: ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.35"); ax.set_global()
    else: ax.set_xlim(-180, 180); ax.set_ylim(-85, 85); ax.grid(alpha=.3)
    ax.legend(loc="lower left", fontsize=8, markerscale=1.4)
    fig.tight_layout(); fig.savefig(f"{OUT}/ar_hierarchy_map_L15.png", dpi=170, bbox_inches="tight")
    print("saved ar_hierarchy_map_L15.png")
if __name__ == "__main__":
    main()
