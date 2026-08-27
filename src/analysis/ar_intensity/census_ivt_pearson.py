import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"
def pearson_cols(A,y,cs=1024):
    yc=y-y.mean(); yn=np.sqrt((yc*yc).sum())
    out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        B=A[:,i:i+cs].astype(np.float64); B=B-B.mean(0)
        out[i:i+cs]=(B*yc[:,None]).sum(0)/(np.sqrt((B*B).sum(0))*yn+1e-12)
    return out
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    C=[]
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); ivt=d["ivt"].astype(float); ok=np.isfinite(ivt)
        C.append(pearson_cols(d["A_mean"][ok],ivt[ok])); del d   # A_mean, max IVT, Pearson
    C=np.vstack(C)
    print(f"\n##### {SAE}  (A_mean, max IVT, Pearson)   max r={C.max():.3f}  per-region max={[round(float(C[i].max()),2) for i in range(4)]}")
    for tau in [0.3,0.4,0.5]:
        act=C>tau; nreg=act.sum(0)
        print(f" -- r>{tau}:  active>=1={int((nreg>=1).sum())}  regional(1)={int((nreg==1).sum())}  two={int((nreg==2).sum())}  global(>=3)={int((nreg>=3).sum())}")
        print(f"    {'region':16}{'Active':>8}{'Regional':>10}{'Global':>8}")
        for ri,r in enumerate(REGIONS):
            a=act[ri]; print(f"    {r:16}{int(a.sum()):8d}{int((a&(nreg==1)).sum()):10d}{int((a&(nreg>=3)).sum()):8d}")
