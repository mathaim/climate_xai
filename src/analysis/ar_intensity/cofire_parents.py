import glob, os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEV,flush=True)
m,c,fmin,frng=load_sae("matry_L8",DEV)
PARENTS=[99,340]; N=8000; F=4096
OUT="/scratch/euh7ys/climate_xai/concept_ivt/cofire_99_340_8k.npz"
files=sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
sel=[files[i] for i in np.linspace(0,len(files)-1,min(N,len(files))).astype(int)]
ce_node=np.zeros(F); ce_ts=np.zeros(F); tot_node=0; tot_ts=0; start=0
both_node={p:np.zeros(F) for p in PARENTS}; both_ts={p:np.zeros(F) for p in PARENTS}
pe_node={p:0 for p in PARENTS}; pe_ts={p:0 for p in PARENTS}
if os.path.exists(OUT):
    z=np.load(OUT); ce_node=z["ce_node"].copy(); ce_ts=z["ce_ts"].copy()
    tot_node=int(z["tot_node"]); tot_ts=int(z["tot_ts"]); start=int(z["ndone"])
    for p in PARENTS:
        both_node[p]=z[f"both_node_{p}"].copy(); both_ts[p]=z[f"both_ts_{p}"].copy()
        pe_node[p]=int(z[f"pe_node_{p}"]); pe_ts[p]=int(z[f"pe_ts_{p}"])
    print(f"resume {start}",flush=True)
def save(nd):
    kw=dict(ce_node=ce_node,ce_ts=ce_ts,tot_node=tot_node,tot_ts=tot_ts,ndone=nd,parents=np.array(PARENTS))
    for p in PARENTS:
        kw[f"both_node_{p}"]=both_node[p]; kw[f"both_ts_{p}"]=both_ts[p]
        kw[f"pe_node_{p}"]=pe_node[p]; kw[f"pe_ts_{p}"]=pe_ts[p]
    np.savez(OUT,**kw)
for j in range(start,len(sel)):
    a=np.load(sel[j],mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    xn=(2*(x-fmin)/frng-1).astype(np.float32) if fmin is not None else x
    with torch.no_grad(): act=encode(m,c["arch"],torch.from_numpy(xn).to(DEV)).cpu().numpy()
    B=act>0.0; fire_ts=B.any(0)
    ce_node+=B.sum(0); ce_ts+=fire_ts; tot_node+=B.shape[0]; tot_ts+=1
    for p in PARENTS:
        pf=B[:,p]; both_node[p]+=(B&pf[:,None]).sum(0); pe_node[p]+=int(pf.sum())
        pfts=bool(pf.any()); both_ts[p]+=fire_ts*pfts; pe_ts[p]+=int(pfts)
    if (j+1)%100==0: save(j+1); print(f"  {j+1}/{len(sel)}",flush=True)
save(len(sel)); print("DONE",flush=True)
