import glob, os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
m,c,fmin,frng=load_sae("matry_L8",DEV)
PARENT=99; N=2500
OUT="/scratch/euh7ys/climate_xai/concept_ivt/containment_99_gt0.npz"
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
sel=[files[i] for i in np.linspace(0,len(files)-1,min(N,len(files))).astype(int)]
ce=np.zeros(4096); both=np.zeros(4096); pe=0; tot=0; start=0
if os.path.exists(OUT):                                   # resume after a timeout
    z=np.load(OUT); ce=z["ce"].copy(); both=z["both"].copy()
    pe=int(z["pe"]); tot=int(z["tot"]); start=int(z["ndone"]); print(f"resume {start}",flush=True)
for j in range(start,len(sel)):
    a=np.load(sel[j],mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()
    B=act>0.0                                             # FIRES = activation > 0 (top-k active)
    pf=B[:,PARENT]
    ce+=B.sum(0); both+=(B&pf[:,None]).sum(0); pe+=int(pf.sum()); tot+=B.shape[0]
    if (j+1)%100==0:
        np.savez(OUT,ce=ce,both=both,pe=pe,tot=tot,ndone=j+1); print(f"  {j+1}/{len(sel)}",flush=True)
contain=both/np.maximum(ce,1); lift=(both/max(tot,1))/((ce/max(tot,1))*(pe/max(tot,1))+1e-12)
np.savez(OUT,ce=ce,both=both,pe=pe,tot=tot,ndone=len(sel),contain=contain,lift=lift); print("DONE",flush=True)
