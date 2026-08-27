import glob, os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR, node_ivt
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
m,c,fmin,frng=load_sae("matry_L8",DEV)
PARENT=99; CHILD=3153; N=8000
OUT="/scratch/euh7ys/climate_xai/concept_ivt/footprint_ivt_3153_8k.npz"
idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
era0=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
NLAT=era0[:,lat_i].astype(float); NLON=era0[:,lon_i].astype(float); NN=len(NLAT)
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy")); sel=[files[i] for i in np.linspace(0,len(files)-1,min(N,len(files))).astype(int)]
NBIN=60; HB=np.linspace(0,1500,NBIN+1); KEYS=["p","c","both","ponly"]
cP=np.zeros(NN); cC=np.zeros(NN); both=np.zeros(NN); sP=np.zeros(NN); sC=np.zeros(NN)
S={k:0.0 for k in KEYS}; SS={k:0.0 for k in KEYS}; Nc={k:0 for k in KEYS}; H={k:np.zeros(NBIN) for k in KEYS}
start=0
if os.path.exists(OUT):
    z=np.load(OUT)
    if "both" in z.files:
        cP=z["cnt_99"].copy(); cC=z["cnt_3153"].copy(); both=z["both"].copy()
        sP=z["sivt_99"].copy(); sC=z["sivt_3153"].copy(); start=int(z["ndone"])
        for k in KEYS: S[k]=float(z[f"S_{k}"]); SS[k]=float(z[f"SS_{k}"]); Nc[k]=int(z[f"N_{k}"]); H[k]=z[f"H_{k}"].copy()
        print(f"resume {start}",flush=True)
def save(nd):
    kw=dict(nlat=NLAT,nlon=NLON,ndone=nd,cnt_99=cP,cnt_3153=cC,both=both,sivt_99=sP,sivt_3153=sC,HB=HB)
    for k in KEYS: kw[f"S_{k}"]=S[k]; kw[f"SS_{k}"]=SS[k]; kw[f"N_{k}"]=Nc[k]; kw[f"H_{k}"]=H[k]
    np.savez(OUT,**kw)
for j in range(start,len(sel)):
    f=sel[j]; a=np.load(f,mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()
    bp=act[:,PARENT]>0.0; bc=act[:,CHILD]>0.0
    cP+=bp; cC+=bc; both+=bp&bc
    ds=f.split("_t")[-1].replace(".npy",""); ef=f"{ERA5_DIR}/era5_inputs_{ds}.npy"
    if os.path.exists(ef):
        era=np.ascontiguousarray(np.load(ef,mmap_mode="r")).astype(np.float64)
        ivt=node_ivt(era,qi,ui,vi,levels)
        sP+=np.where(bp,ivt,0.0); sC+=np.where(bc,ivt,0.0)
        msk={"p":bp,"c":bc,"both":bp&bc,"ponly":bp&~bc}
        for k in KEYS:
            v=ivt[msk[k]]
            if v.size: S[k]+=v.sum(); SS[k]+=float((v*v).sum()); Nc[k]+=v.size; H[k]+=np.histogram(v,HB)[0]
    if (j+1)%100==0: save(j+1); print(f"  {j+1}/{len(sel)}",flush=True)
save(len(sel)); print("DONE",flush=True)
