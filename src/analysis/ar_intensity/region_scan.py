"""Scan all concepts for spatial containment: fraction of firings in the top-1% of mesh nodes (global footprint)."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
SAE = "matry_L8"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "300"))
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/region_scan.npz"
conv = lambda x: x - 360 if x > 180 else x
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i]; nlon = np.array([conv(x) for x in era0[:, lon_i]]); nnode = era0.shape[0]
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(0); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    fc = np.zeros((nnode, 4096), np.float32); cnt = 0
    for i, f in enumerate(sel):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()
        fc += (acts > THRESH); cnt += 1
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    tot = fc.sum(0); ntop = max(1, nnode // 100)
    conc = np.partition(fc, -ntop, axis=0)[-ntop:, :].sum(0) / np.maximum(tot, 1)   # frac in top-1% nodes
    peak = fc.argmax(0)
    np.savez(OUT, conc=conc, tot=tot, peaklat=nlat[peak], peaklon=nlon[peak])
    order = [c2 for c2 in np.argsort(-conc) if tot[c2] > 200][:30]
    print(f"{'concept':>7} {'conc':>5} {'fires':>8} {'peak lat':>9} {'peak lon':>9}")
    for c2 in order:
        print(f"{c2:>7} {conc[c2]:>5.2f} {int(tot[c2]):>8} {nlat[peak[c2]]:>9.1f} {nlon[peak[c2]]:>9.1f}")
    for ref in [99, 3153, 3483]:
        print(f" ref {ref}: conc={conc[ref]:.2f} fires={int(tot[ref])} peak={nlat[peak[ref]]:.0f},{nlon[peak[ref]]:.0f}")
if __name__ == "__main__":
    main()
