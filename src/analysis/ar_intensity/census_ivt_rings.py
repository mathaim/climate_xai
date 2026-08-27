import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; BINS=[256,512,1024,2048]; idx=np.arange(4096); shell=np.digitize(idx,BINS)
def spearman_cols(A,y,cs=512):
    ry=np.argsort(np.argsort(y)).astype(np.float64); ry-=ry.mean(); ryn=np.sqrt((ry*ry).sum())
    out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        R=np.argsort(np.argsort(A[:,i:i+cs].astype(np.float64),0),0).astype(np.float64); R-=R.mean(0)
        out[i:i+cs]=(R*ry[:,None]).sum(0)/(np.sqrt((R*R).sum(0))*ryn+1e-12)
    return out
for SAE,stem in [("matry_L8","track_matry"),("plain_L8","track_pool")]:
    C=[]
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); ivt=d["ivt"].astype(float); ok=np.isfinite(ivt)
        C.append(spearman_cols(d["A_max"][ok],ivt[ok])); del d
    C=np.vstack(C)
    print(f"\n##### {SAE}   (core=shell0 is 6.25% of latents; shell sizes 256/256/512/1024/2048)")
    for tau in [0.2,0.3]:
        nreg=(C>tau).sum(0)
        print(f" -- corr>{tau} --")
        for lab,mask in [("GLOBAL (>=3)",nreg>=3),("two",nreg==2),("REGIONAL(1)",nreg==1)]:
            n=int(mask.sum()); cnt=[int((mask&(shell==s)).sum()) for s in range(5)]
            print(f"   {lab:12} n={n:3d}  shells[core..out] {cnt}  core%={100*cnt[0]/max(n,1):4.1f}  med_idx={int(np.median(idx[mask])) if n else 0}")
        g=np.where(nreg>=3)[0]
        print(f"   global ids(shell): "+", ".join(f"{c}(s{shell[c]})" for c in g))
