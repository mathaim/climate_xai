"""Local/global AR census from IVT-tracking: a concept 'tracks ARs in region r' if
spearman(A_max, IVT)_r > tau. Regional=tracks in exactly 1 region, Global=>=3. Both archs."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"
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
    print(f"\n##### {SAE}")
    for tau in [0.2,0.3,0.4]:
        act=C>tau; nreg=act.sum(0)
        print(f"-- corr>{tau}:  concepts tracking >=1 region={int((nreg>=1).sum())}  |  regional(1)={int((nreg==1).sum())}  two(2)={int((nreg==2).sum())}  global(>=3)={int((nreg>=3).sum())}")
        print(f"   {'region':16}{'Active':>8}{'Regional':>10}{'Global':>8}")
        for ri,r in enumerate(REGIONS):
            a=act[ri]; print(f"   {r:16}{int(a.sum()):8d}{int((a&(nreg==1)).sum()):10d}{int((a&(nreg>=3)).sum()):8d}")
