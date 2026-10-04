import os, glob, numpy as np, torch
from analysis.lib.sae_features import load_sae, encode
from analysis.lib.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
m,c,fmin,frng=load_sae("matry_L8",DEV)
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy")); N=8000; THR=0.1; CC=99
sel=[files[i] for i in np.linspace(0,len(files)-1,N).astype(int)]
def acc(): return dict(sx=0.0,sxx=0.0,sy=0.0,syy=0.0,sxy=0.0,n=0)
def add(d,x,y):
    d['sx']+=x.sum(); d['sxx']+=(x*x).sum(); d['sy']+=y.sum(); d['syy']+=(y*y).sum(); d['sxy']+=(x*y).sum(); d['n']+=len(x)
def r_of(d):
    n=d['n']; num=n*d['sxy']-d['sx']*d['sy']
    den=np.sqrt(max(n*d['sxx']-d['sx']**2,1e-9)*max(n*d['syy']-d['sy']**2,1e-9)); return num/den
A=acc(); F=acc()
for j,f in enumerate(sel):
    ds=f.split("_t")[-1].replace(".npy",""); ef=f"{ERA5_DIR}/era5_inputs_{ds}.npy"
    if not os.path.exists(ef): continue
    a=np.load(f,mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()[:,CC].astype(np.float64)
    era=np.ascontiguousarray(np.load(ef,mmap_mode="r")).astype(np.float64)
    iv=node_ivt(era,qi,ui,vi,levels).astype(np.float64)
    add(A,act,iv)
    mfire=act>THR
    if mfire.sum()>1: add(F,act[mfire],iv[mfire])
    if (j+1)%1000==0: print(f"  {j+1}/{N}",flush=True)
print(f"\nconcept 99 vs node-level IVT, pooled over 1979-2017 ({N} sampled timesteps):",flush=True)
print(f"  ALL node-timesteps:        r = {r_of(A):.3f}  (n={A['n']})",flush=True)
print(f"  where 99 fires (act>{THR}): r = {r_of(F):.3f}  (n={F['n']})",flush=True)
print("DONE",flush=True)
