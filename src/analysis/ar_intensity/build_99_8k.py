import os, glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
QI,UI,VI=set(map(int,qi)),set(map(int,ui)),set(map(int,vi))
def kind(k):
    if k in QI: return f"specific humidity L{levels[list(qi).index(k)]:.0f}"
    if k in UI: return "u wind"
    if k in VI: return "v wind"
    return "other"
era0=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
NLAT=era0[:,lat_i].astype(float); NLON=era0[:,lon_i].astype(float); NN=len(NLAT); CH=era0.shape[1]
m,c,fmin,frng=load_sae("matry_L8",DEV)
CC=[99,3392,1454,2722]; THRESH=0.1
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy")); N=8000
sel=[files[i] for i in np.linspace(0,len(files)-1,N).astype(int)]
cnt={cc:np.zeros(NN,dtype=np.int32) for cc in CC}
sx={cc:0.0 for cc in CC}; sxx={cc:0.0 for cc in CC}; n={cc:0 for cc in CC}
sy={cc:np.zeros(CH) for cc in CC}; syy={cc:np.zeros(CH) for cc in CC}; sxy={cc:np.zeros(CH) for cc in CC}
for j,f in enumerate(sel):
    a=np.load(f,mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()
    for cc in CC: cnt[cc]+=(act[:,cc]>THRESH)
    if j%5==0:
        ds=f.split("_t")[-1].replace(".npy",""); ef=f"{ERA5_DIR}/era5_inputs_{ds}.npy"
        if os.path.exists(ef):
            era=np.ascontiguousarray(np.load(ef,mmap_mode="r")).astype(np.float64)
            for cc in CC:
                mask=act[:,cc]>THRESH
                if mask.sum()>1:
                    xc=act[mask,cc].astype(np.float64); Y=era[mask]
                    sx[cc]+=xc.sum(); sxx[cc]+=(xc*xc).sum(); n[cc]+=int(mask.sum())
                    sy[cc]+=Y.sum(0); syy[cc]+=(Y*Y).sum(0); sxy[cc]+=(xc[:,None]*Y).sum(0)
    if (j+1)%1000==0: print(f"  {j+1}/{N}",flush=True)
np.savez("/scratch/euh7ys/climate_xai/concept_ivt/footprint_99_8k.npz",nlat=NLAT,nlon=NLON,nsteps=N,**{f"cnt_{cc}":cnt[cc] for cc in CC})
print("\n=== 99-family channel correlations (activation vs input, pooled over active nodes) ===",flush=True)
for cc in CC:
    Nn=n[cc]; num=Nn*sxy[cc]-sx[cc]*sy[cc]
    den=np.sqrt(np.maximum(Nn*sxx[cc]-sx[cc]**2,1e-9)*np.maximum(Nn*syy[cc]-sy[cc]**2,1e-9))
    r=num/den; top=np.argsort(-np.abs(r))[:6]
    print(f"\nconcept {cc} (n={Nn} active-node samples):",flush=True)
    for k in top: print(f"   ch{k:3d} r={r[k]:+.3f}  {kind(k)}",flush=True)
print("DONE",flush=True)
