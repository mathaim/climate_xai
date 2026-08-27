import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; BINS=[256,512,1024,2048]; shell=np.digitize(np.arange(4096),BINS); RL=list(REGIONS)
def pcols(A,y,cs=1024):
    yc=y-y.mean(); yn=np.sqrt((yc*yc).sum()); out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        B=A[:,i:i+cs].astype(np.float64); B=B-B.mean(0)
        out[i:i+cs]=(B*yc[:,None]).sum(0)/(np.sqrt((B*B).sum(0))*yn+1e-12)
    return out
for SAE,stem in [("matry","track_matry"),("plain","track_pool")]:
    C=[]
    for r in RL:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); ivt=d["ivt"].astype(float); ok=np.isfinite(ivt)
        C.append(pcols(d["A_mean"][ok],ivt[ok])); del d
    C=np.vstack(C); nreg=(C>0.3).sum(0)
    print(f"\n{SAE}: ring  n   AR(%)    reg  two  glob   %AR  %glob-of-AR")
    for s in range(5):
        m=shell==s; n=int(m.sum()); ar=int((m&(nreg>=1)).sum())
        reg=int((m&(nreg==1)).sum()); two=int((m&(nreg==2)).sum()); gl=int((m&(nreg>=3)).sum())
        print(f"  {s}  {n:5d}  {ar:3d}   {reg:4d} {two:4d} {gl:4d}   {100*ar/n:4.1f}  {100*gl/max(ar,1):4.1f}")
