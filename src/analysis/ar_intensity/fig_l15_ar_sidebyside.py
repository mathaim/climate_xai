"""Side-by-side L8 vs L15 for the 99 family (strong firings, global). L8: core 99 + children
in their basins. L15: 99 and each child's BEST L15 match (near-zero Jaccard) landing in
unrelated basins -> the AR family does not survive to L15. No descriptive title."""
import os, glob, numpy as np, torch, datetime as DT, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
N = 200; THRESH = 0.1; OUT = "/scratch/euh7ys/climate_xai/plots"; conv = lambda x: x - 360 if x > 180 else x
CHILD = [(1454, 864, "#c0392b", "1454 $\\to$ 864  (J=0.10)"),
         (3392, 3735, "#e67e22", "3392 $\\to$ 3735  (J=0.03)"),
         (2722, 4036, "#2ca02c", "2722 $\\to$ 4036  (J=0.01)")]
L8C = [99] + [c[0] for c in CHILD]; L15C = [111] + [c[1] for c in CHILD]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def collect(files, sae, lats, nlon, nlat):
    m, c, fmin, frng = load_sae(sae, "cpu"); P = {l: {"lon": [], "lat": [], "act": []} for l in lats}
    for f in files:
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for l in lats:
            fire = acts[:, l] > THRESH
            if fire.any(): P[l]["lon"].extend(nlon[fire]); P[l]["lat"].extend(nlat[fire]); P[l]["act"].extend(acts[fire, l])
    return P
def strong(P, l):
    lo = np.array(P[l]["lon"]); la = np.array(P[l]["lat"]); ac = np.array(P[l]["act"])
    if len(ac) == 0: return lo, la
    s = ac >= np.quantile(ac, 0.99); return lo[s], la[s]
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i]; nlon = np.array([conv(x) for x in era0[:, lon_i]])
    _, c8, _, _ = load_sae("matry_L8", "cpu"); _, c15, _, _ = load_sae("matry_L15", "cpu")
    dtmap = lambda d: {pdt(os.path.basename(f)): f for f in glob.glob(f"{d}/layer*_*.npy")}
    f8, f15 = dtmap(c8["act"]), dtmap(c15["act"]); shared = sorted(set(f8) & set(f15))
    rng = np.random.default_rng(0); sel = [shared[i] for i in rng.choice(len(shared), min(N, len(shared)), replace=False)]
    P8 = collect([f8[d] for d in sel], "matry_L8", L8C, nlon, nlat)
    P15 = collect([f15[d] for d in sel], "matry_L15", L15C, nlon, nlat)
    try: import cartopy.crs as ccrs, cartopy.feature as cfeature; HAVE = True
    except Exception: HAVE = False
    proj = {"projection": ccrs.PlateCarree()} if HAVE else {}; tk = {"transform": ccrs.PlateCarree()} if HAVE else {}
    fig = plt.figure(figsize=(16, 5))
    axL = fig.add_subplot(1, 2, 1, **proj); axL.set_title("L8")
    lo, la = strong(P8, 99); axL.scatter(lo, la, s=6, c="0.55", alpha=0.5, edgecolor="none", label="99 core", **tk)
    for l8, l15, col, _ in CHILD:
        lo, la = strong(P8, l8); axL.scatter(lo, la, s=18, c=col, edgecolor="k", lw=0.2, label=f"{l8}", **tk)
    axR = fig.add_subplot(1, 2, 2, **proj); axR.set_title("L15")
    lo, la = strong(P15, 111); axR.scatter(lo, la, s=6, c="0.55", alpha=0.5, edgecolor="none", label="99 best match (111, J=0.26)", **tk)
    for l8, l15, col, lab in CHILD:
        lo, la = strong(P15, l15); axR.scatter(lo, la, s=18, c=col, edgecolor="k", lw=0.2, label=lab, **tk)
    for ax in (axL, axR):
        if HAVE: ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4"); ax.set_global()
        else: ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=.3)
        ax.legend(loc="lower left", fontsize=8, markerscale=1.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/l15_ar_sidebyside.png", dpi=170, bbox_inches="tight"); print("saved l15_ar_sidebyside.png")
if __name__ == "__main__":
    main()
