"""Temporal + spatial containment for the two families (matry_L8). Optimized (GPU-masked, only the
CC columns move to CPU) and checkpointed (resumes on restart).
 spatial  S = |nodes(child) & nodes(parent)| / |nodes(child)|
 temporal P(parent|child) at node-event AND timestep level, each with marginal P(parent) and lift
 breadth  = distinct firing nodes (node set). firing = activation > THRESH (default 0)."""
import os, glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
N=int(os.environ.get("NMAX","8000")); THRESH=float(os.environ.get("THRESH","0.0"))
DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT="/scratch/euh7ys/climate_xai/plots"; CKPT=f"{OUT}/containment.ckpt.npz"
FAM={340:[3481,3948,3675], 99:[1454,3392,2722]}; CC=[c for p in FAM for c in [p]+FAM[p]]; CH=[x for p in FAM for x in FAM[p]]
def main():
    print("dev",DEV,"N",N,"THRESH",THRESH,flush=True)
    m,c,fmin,frng=load_sae("matry_L8",DEV); CCt=torch.as_tensor(CC,device=DEV)
    files=sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    sel=files if N>=len(files) else [files[i] for i in np.linspace(0,len(files)-1,N).astype(int)]
    ne={cc:0 for cc in CC}; cofNE={x:0 for x in CH}; cofTS={x:0 for x in CH}; cTS={x:0 for x in CH}
    pTS={p:0 for p in FAM}; ngrid=0; nts=0; start=0; ncnt=None
    if os.path.exists(CKPT):
        z=np.load(CKPT); start=int(z["j"][0]); ngrid=int(z["ngrid"][0]); nts=int(z["nts"][0])
        ncnt={cc:z[f"ncnt_{cc}"].copy() for cc in CC}
        for cc in CC: ne[cc]=int(z[f"ne_{cc}"][0])
        for x in CH: cofNE[x]=int(z[f"cofNE_{x}"][0]); cofTS[x]=int(z[f"cofTS_{x}"][0]); cTS[x]=int(z[f"cTS_{x}"][0])
        for p in FAM: pTS[p]=int(z[f"pTS_{p}"][0])
        print("RESUMED at",start,flush=True)
    def ckpt(j):
        d={"j":np.array([j]),"ngrid":np.array([ngrid]),"nts":np.array([nts])}
        for cc in CC: d[f"ncnt_{cc}"]=ncnt[cc]; d[f"ne_{cc}"]=np.array([ne[cc]])
        for x in CH: d[f"cofNE_{x}"]=np.array([cofNE[x]]); d[f"cofTS_{x}"]=np.array([cofTS[x]]); d[f"cTS_{x}"]=np.array([cTS[x]])
        for p in FAM: d[f"pTS_{p}"]=np.array([pTS[p]])
        np.savez(CKPT,**d)
    for k in range(start,len(sel)):
        a=np.load(sel[k],mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        if fmin is not None: x=(2.0*(x-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad():
            av=encode(m,c["arch"],torch.from_numpy(x).to(DEV)); fb=(av[:,CCt]>THRESH).cpu().numpy()
        if ncnt is None: ncnt={cc:np.zeros(fb.shape[0],np.int64) for cc in CC}
        fire={cc:fb[:,i] for i,cc in enumerate(CC)}
        for cc in CC: ncnt[cc]+=fire[cc]; ne[cc]+=int(fire[cc].sum())
        for p,chs in FAM.items():
            pa=bool(fire[p].any()); pTS[p]+=pa
            for x2 in chs:
                cofNE[x2]+=int((fire[x2]&fire[p]).sum()); ca=bool(fire[x2].any()); cTS[x2]+=ca; cofTS[x2]+=(ca and pa)
        ngrid+=fb.shape[0]; nts+=1
        if (k+1)%1000==0: ckpt(k+1); print(f"  {k+1}/{len(sel)} ckpt",flush=True)
    print(f"\nfiring>{THRESH}  timesteps={nts}  node-events={ngrid}\n")
    for p,chs in FAM.items():
        Ppne=ne[p]/ngrid; Ppts=pTS[p]/nts; pn=ncnt[p]>0
        print(f"=== parent {p}: distinct_nodes={int(pn.sum())}  P(par)_ne={Ppne:.4g}  P(par)_ts={Ppts:.3f} ===")
        print(f"{'child':>6}{'nodes':>7}{'events':>10}{'Pne(par|ch)':>12}{'liftNE':>8}{'Pts(par|ch)':>13}{'liftTS':>8}{'spatialS':>10}")
        for x2 in chs:
            Tne=cofNE[x2]/max(ne[x2],1); Tts=cofTS[x2]/max(cTS[x2],1); cn=ncnt[x2]>0; S=(cn&pn).sum()/max(cn.sum(),1)
            print(f"{x2:>6}{int(cn.sum()):>7}{ne[x2]:>10}{Tne:>12.3f}{(Tne/Ppne if Ppne else 0):>8.0f}{Tts:>13.3f}{(Tts/Ppts if Ppts else 0):>8.1f}{S:>10.3f}")
    if os.path.exists(CKPT): os.remove(CKPT)
if __name__=="__main__": main()
