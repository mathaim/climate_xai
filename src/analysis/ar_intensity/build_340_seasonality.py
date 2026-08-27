import glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, region_node_setup
DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device", DEV, flush=True)
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
nodes = region_node_setup()["W_S_America"]["nodes"]
m, c, fmin, frng = load_sae("matry_L8", DEV)
CC = [340, 3481, 3948, 3675]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")
files = sorted(glob.glob(f"{c['act']}/layer*_*.npy")); N = 8000
sel = [files[i] for i in np.linspace(0, len(files)-1, N).astype(int)]
mon, yr, A = [], [], {cc: [] for cc in CC}
for j, f in enumerate(sel):
    dt = pdt(f.split("/")[-1]); mon.append(dt.month); yr.append(dt.year)
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a[nodes]).astype(np.float32).reshape(len(nodes), -1)
    xn = (2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act = encode(m, c["arch"], torch.from_numpy(xn).to(DEV)).cpu().numpy()
    for cc in CC: A[cc].append(float(act[:, cc].mean()))
    if (j+1) % 1000 == 0: print(f"  {j+1}/{N}", flush=True)
mon = np.array(mon); yr = np.array(yr); A = {cc: np.array(A[cc]) for cc in CC}
np.savez("/scratch/euh7ys/climate_xai/concept_ivt/season_340.npz", month=mon, year=yr, **{f"a_{cc}": A[cc] for cc in CC})
MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
print("\nMonthly mean activation:")
print("     " + "".join(f"{cc:>9}" for cc in CC))
for mm in range(1, 13):
    s = mon == mm
    print(f"{MON[mm-1]:>4} " + "".join(f"{A[cc][s].mean():9.4f}" for cc in CC))
print("\n340 by 4-year block:")
for y in range(int(yr.min()), int(yr.max())+1, 4):
    s = (yr>=y)&(yr<y+4)
    if s.any(): print(f"  {y}-{y+3}: {A[340][s].mean():.4f}  (n={s.sum()})")
print("DONE", flush=True)
