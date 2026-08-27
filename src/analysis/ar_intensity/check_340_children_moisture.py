"""Same channel-correlation test for the 340 family: parent + three children (matry-L8).
Confirms whether the whole family is the dry (inverse-humidity) coastal regime."""
import os, glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR, region_node_setup, node_ivt
DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device", DEV, flush=True)
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
QI, UI, VI = set(int(x) for x in qi), set(int(x) for x in ui), set(int(x) for x in vi)
def kind(k):
    if k in QI: return f"q L{levels[list(qi).index(k)]:.0f}"
    if k in UI: return "u wind"
    if k in VI: return "v wind"
    return "other"
nodes = region_node_setup()["W_S_America"]["nodes"]
m, c, fmin, frng = load_sae("matry_L8", DEV)
CC = [340, 3481, 3948, 3675]
files = sorted(glob.glob(f"{c['act']}/layer*_*.npy")); N = 1200
sel = [files[i] for i in np.linspace(0, len(files)-1, N).astype(int)]
A = {cc: [] for cc in CC}; X, IV = [], []
for j, f in enumerate(sel):
    ds = f.split("_t")[-1].replace(".npy", ""); ef = f"{ERA5_DIR}/era5_inputs_{ds}.npy"
    if not os.path.exists(ef): continue
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a[nodes]).astype(np.float32).reshape(len(nodes), -1)
    xn = (2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act = encode(m, c["arch"], torch.from_numpy(xn).to(DEV)).cpu().numpy()
    for cc in CC: A[cc].append(float(act[:, cc].mean()))
    era = np.ascontiguousarray(np.load(ef, mmap_mode="r")[nodes]).astype(np.float32)
    X.append(era.mean(0)); IV.append(float(node_ivt(era, qi, ui, vi, levels).mean()))
    if (j+1) % 400 == 0: print(f"  {j+1}/{len(sel)}", flush=True)
X = np.array(X); IV = np.array(IV)
for cc in CC:
    av = np.array(A[cc]); r = np.array([np.corrcoef(av, X[:, k])[0,1] if X[:, k].std()>0 else 0.0 for k in range(X.shape[1])])
    top = np.argsort(-np.abs(r))[:5]
    print(f"\n=== {cc} ===  std/mean {av.std()/max(av.mean(),1e-9):.2f}   corr(IVT) {np.corrcoef(av,IV)[0,1]:+.3f}")
    for k in top: print(f"   ch{k:3d} r={r[k]:+.3f}  {kind(k)}")
