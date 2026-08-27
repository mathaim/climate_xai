import glob, os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
m,c,fmin,frng=load_sae("matry_L8",DEV)
CC=[99,3153]; N=8000
OUT="/scratch/euh7ys/climate_xai/concept_ivt/footprint_3153_8k.npz"
idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
era0=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
NLAT=era0[:,lat_i].astype(float); NLON=era0[:,lon_i].astype(float); NN=len(NLAT)
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy")); sel=[files[i] for i in np.linspace(0,len(files)-1,min(N,len(files))).astype(int)]
cnt={cc:np.zeros(NN) for cc in CC}; start=0
if os.path.exists(OUT):
    z=np.load(OUT); start=int(z["ndone"]); cnt={cc:z[f"cnt_{cc}"].copy() for cc in CC}; print(f"resume {start}",flush=True)
def save(nd): np.savez(OUT,nlat=NLAT,nlon=NLON,ndone=nd,**{f"cnt_{cc}":cnt[cc] for cc in CC})
for j in range(start,len(sel)):
    a=np.load(sel[j],mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()
    B=act>0.0
    for cc in CC: cnt[cc]+=B[:,cc]
    if (j+1)%200==0: save(j+1); print(f"  {j+1}/{len(sel)}",flush=True)
save(len(sel)); print("DONE",flush=True)
