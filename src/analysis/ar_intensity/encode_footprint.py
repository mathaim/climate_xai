"""Strong-firing footprint of a region concept + local IVT & parent(99) co-firing at each firing node.
Feeds the 'where x how-strong' figure: (a) where CC fires, (c) at what local IVT (vs global)."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
SAE = "matry_L8"
CC = int(os.environ.get("CC", "3153"))        # region-contained concept
PARENT = int(os.environ.get("PARENT", "99"))  # global intensity concept
THRESH = 0.1
N = int(os.environ.get("GLOBAL_N", "500"))
OUT = os.environ.get("OUT", f"/scratch/euh7ys/climate_xai/concept_ivt/footprint_{CC}.npz")
conv = lambda x: x - 360 if x > 180 else x
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(float); nlon = np.array([conv(x) for x in era0[:, lon_i]], float); nnode = era0.shape[0]
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(1); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    ev_node, ev_act, ev_ivt, ev_pval = [], [], [], []
    full_count = np.zeros(nnode); used = 0
    for i, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()
        aC, aP = acts[:, CC], acts[:, PARENT]
        mk = np.where(aC > THRESH)[0]; used += 1; full_count[mk] += 1
        ev_node.append(mk); ev_act.append(aC[mk]); ev_ivt.append(iv[mk]); ev_pval.append(aP[mk])
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(sel)} events={sum(len(e) for e in ev_node)}", flush=True)
    ev_node = np.concatenate(ev_node); ev_act = np.concatenate(ev_act)
    ev_ivt = np.concatenate(ev_ivt); ev_pval = np.concatenate(ev_pval)
    thr = float(np.quantile(ev_act, 0.99)); strong = ev_act >= thr
    strong_count = np.bincount(ev_node[strong], minlength=nnode).astype(float)
    np.savez(OUT, nlat=nlat, nlon=nlon, full_count=full_count, strong_count=strong_count,
             ev_node=ev_node, ev_act=ev_act, ev_ivt=ev_ivt, ev_pval=ev_pval,
             thr=thr, used=used, cc=CC, parent=PARENT)
    la, lo = nlat[ev_node[strong]], nlon[ev_node[strong]]
    print(f"used={used} events={len(ev_act)} strong(top1%)={int(strong.sum())} thr={thr:.3f}", flush=True)
    print(f"medIVT: strong={np.median(ev_ivt[strong]):.0f}  all-firings={np.median(ev_ivt):.0f}", flush=True)
    print(f"parent {PARENT} act at CC firings: strong-mean={ev_pval[strong].mean():.3f}  all-mean={ev_pval.mean():.3f}", flush=True)
    print(f"strong-firing loc: median lat={np.median(la):.1f} lon={np.median(lo):.1f}  "
          f"lat[10,90]%=[{np.percentile(la,10):.0f},{np.percentile(la,90):.0f}] "
          f"lon[10,90]%=[{np.percentile(lo,10):.0f},{np.percentile(lo,90):.0f}]", flush=True)
if __name__ == "__main__":
    main()
