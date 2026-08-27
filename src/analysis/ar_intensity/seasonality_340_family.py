import glob, os, re, numpy as np, torch
from datetime import datetime
from src.analysis.ar_intensity.sae_features import load_sae, encode
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
m,c,fmin,frng=load_sae("matry_L8",DEV)
CC=[340,3757,2858,3112,2474,3495,3481,3675,3700,1399,1622,3126,3948]; N=8000
OUT="/scratch/euh7ys/climate_xai/concept_ivt/seasonality_340_family_8k.npz"
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
sel=[files[i] for i in np.linspace(0,len(files)-1,min(N,len(files))).astype(int)]
pat=re.compile(r"_t(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})")
dts=[datetime(*map(int,pat.search(f).groups())) for f in sel]
months=np.array([d.month for d in dts]); doy=np.array([d.timetuple().tm_yday for d in dts]); years=np.array([d.year for d in dts])
M=np.zeros((len(sel),len(CC)),dtype=np.int32); start=0
if os.path.exists(OUT):
    z=np.load(OUT); start=int(z["ndone"]); M=z["M"].copy(); print(f"resume {start}",flush=True)
def save(nd): np.savez(OUT,M=M,cc=np.array(CC),months=months,doy=doy,years=years,ndone=nd)
for j in range(start,len(sel)):
    a=np.load(sel[j],mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()
    B=act>0.0
    for k,cc in enumerate(CC): M[j,k]=int(B[:,cc].sum())
    if (j+1)%200==0: save(j+1); print(f"  {j+1}/{len(sel)}",flush=True)
save(len(sel)); print("DONE",flush=True)
