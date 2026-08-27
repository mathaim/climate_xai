import os, glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
qi=list(map(int,qi)); ui=list(map(int,ui)); vi=list(map(int,vi)); levels=[float(l) for l in levels]
m,c,fmin,frng=load_sae("matry_L8",DEV)
CC=[340,3757,2858,3112,2474,3495,3481,3675,3700,1399,1622,3126,3948]; N=8000
OUT="/scratch/euh7ys/climate_xai/concept_ivt/chancorr_340.npz"
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy")); sel=[files[i] for i in np.linspace(0,len(files)-1,min(N,len(files))).astype(int)]
NCH=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]).shape[1]
sx={k:0.0 for k in CC}; sxx={k:0.0 for k in CC}; n={k:0 for k in CC}
sy={k:np.zeros(NCH) for k in CC}; syy={k:np.zeros(NCH) for k in CC}; sxy={k:np.zeros(NCH) for k in CC}
start=0
if os.path.exists(OUT):
    z=np.load(OUT); start=int(z["ndone"])
    for k in CC:
        sx[k]=float(z[f"sx_{k}"]); sxx[k]=float(z[f"sxx_{k}"]); n[k]=int(z[f"n_{k}"])
        sy[k]=z[f"sy_{k}"].copy(); syy[k]=z[f"syy_{k}"].copy(); sxy[k]=z[f"sxy_{k}"].copy()
    print(f"resume {start}",flush=True)
def save(nd):
    kw={"ndone":nd,"cc":np.array(CC),"qi":np.array(qi),"ui":np.array(ui),"vi":np.array(vi),"levels":np.array(levels)}
    for k in CC:
        kw[f"sx_{k}"]=sx[k]; kw[f"sxx_{k}"]=sxx[k]; kw[f"n_{k}"]=n[k]
        kw[f"sy_{k}"]=sy[k]; kw[f"syy_{k}"]=syy[k]; kw[f"sxy_{k}"]=sxy[k]
    np.savez(OUT,**kw)
for j in range(start,len(sel)):
    f=sel[j]; a=np.load(f,mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()
    ds=f.split("_t")[-1].replace(".npy",""); ef=f"{ERA5_DIR}/era5_inputs_{ds}.npy"
    if not os.path.exists(ef): continue
    era=np.ascontiguousarray(np.load(ef,mmap_mode="r")).astype(np.float64)
    for k in CC:
        mm=act[:,k]>0.0
        if mm.sum()>1:
            xc=act[mm,k].astype(np.float64); Y=era[mm]
            sx[k]+=xc.sum(); sxx[k]+=(xc*xc).sum(); n[k]+=int(mm.sum())
            sy[k]+=Y.sum(0); syy[k]+=(Y*Y).sum(0); sxy[k]+=(xc[:,None]*Y).sum(0)
    if (j+1)%200==0: save(j+1); print(f"  {j+1}/{len(sel)}",flush=True)
save(len(sel)); print("DONE",flush=True)
