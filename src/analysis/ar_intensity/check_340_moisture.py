"""Does matryoshka-L8 concept 340 track moisture? Correlate its W-S-America activation over time
against every ERA5 input channel (and IVT), then rank. Uses the CORRECT (matry) SAE."""
import os, glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR, region_node_setup, node_ivt
DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device", DEV, flush=True)
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
QI, UI, VI = set(int(x) for x in qi), set(int(x) for x in ui), set(int(x) for x in vi)
def kind(k):
    if k in QI: return f"specific humidity  L{levels[list(qi).index(k)]:.0f}"
    if k in UI: return "u wind"
    if k in VI: return "v wind"
    return "other (z/T/w/surface/static)"
nodes = region_node_setup()["W_S_America"]["nodes"]
m, c, fmin, frng = load_sae("matry_L8", DEV)
files = sorted(glob.glob(f"{c['act']}/layer*_*.npy")); N = 1200
sel = [files[i] for i in np.linspace(0, len(files)-1, N).astype(int)]
A, X, IV = [], [], []
for j, f in enumerate(sel):
    ds = f.split("_t")[-1].replace(".npy", ""); ef = f"{ERA5_DIR}/era5_inputs_{ds}.npy"
    if not os.path.exists(ef): continue
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a[nodes]).astype(np.float32).reshape(len(nodes), -1)
    xn = (2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act = encode(m, c["arch"], torch.from_numpy(xn).to(DEV)).cpu().numpy()[:, 340]
    era = np.ascontiguousarray(np.load(ef, mmap_mode="r")[nodes]).astype(np.float32)
    A.append(float(act.mean())); X.append(era.mean(0)); IV.append(float(node_ivt(era, qi, ui, vi, levels).mean()))
    if (j+1) % 300 == 0: print(f"  {j+1}/{len(sel)}", flush=True)
A = np.array(A); X = np.array(X); IV = np.array(IV)
print(f"\n340 WSA activation:  mean {A.mean():.4f}  std {A.std():.4f}  frac>0.01 {(A>0.01).mean()*100:.1f}%")
print(f"corr(340, IVT) = {np.corrcoef(A, IV)[0,1]:+.3f}")
r = np.array([np.corrcoef(A, X[:, k])[0,1] if X[:, k].std() > 0 else 0.0 for k in range(X.shape[1])])
print("\nTop 15 input channels by |corr| with 340 activation:")
for k in np.argsort(-np.abs(r))[:15]:
    print(f"  ch{k:3d}  r={r[k]:+.3f}   {kind(k)}")
