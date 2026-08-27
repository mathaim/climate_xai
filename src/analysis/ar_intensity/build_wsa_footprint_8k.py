"""Per-node firing counts for the 340 family (matry-L8) over 8000 evenly-strided timesteps, global."""
import glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device", DEV, flush=True)
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
nlat = era0[:, lat_i].astype(float); nlon = era0[:, lon_i].astype(float); NN = len(nlat)
m, c, fmin, frng = load_sae("matry_L8", DEV)
files = sorted(glob.glob(f"{c['act']}/layer*_*.npy")); N = 8000
sel = [files[i] for i in np.linspace(0, len(files)-1, N).astype(int)]
CC = [340, 3481, 3948, 3675]; THRESH = 0.1
cnt = {cc: np.zeros(NN, dtype=np.int32) for cc in CC}
for j, f in enumerate(sel):
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
    xn = (2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act = encode(m, c["arch"], torch.from_numpy(xn).to(DEV)).cpu().numpy()
    for cc in CC: cnt[cc] += (act[:, cc] > THRESH)
    if (j+1) % 1000 == 0: print(f"  {j+1}/{N}", flush=True)
np.savez("/scratch/euh7ys/climate_xai/concept_ivt/characterize_wsa_8k.npz",
         nlat=nlat, nlon=nlon, nsteps=N, **{f"cnt_{cc}": cnt[cc] for cc in CC})
print("DONE  total firings:", {cc: int(cnt[cc].sum()) for cc in CC}, flush=True)
