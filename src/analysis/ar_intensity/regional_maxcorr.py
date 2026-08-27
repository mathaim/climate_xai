import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; RL=list(REGIONS)
def spearman_cols(A,y,cs=512):
    ry=np.argsort(np.argsort(y)).astype(np.float64); ry-=ry.mean(); ryn=np.sqrt((ry*ry).sum())
    out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        R=np.argsort(np.argsort(A[:,i:i+cs].astype(np.float64),0),0).astype(np.float64); R-=R.mean(0)
        out[i:i+cs]=(R*ry[:,None]).sum(0)/(np.sqrt((R*R).sum(0))*ryn+1e-12)
    return out
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    C=[]
    for r in RL:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); ivt=d["ivt"].astype(float); ok=np.isfinite(ivt)
        C.append(spearman_cols(d["A_max"][ok],ivt[ok])); del d
    C=np.vstack(C); nreg=(C>0.3).sum(0); reg=nreg==1; glob=nreg>=3; mx=C.max(0)
    print(f"\n### {SAE}")
    print(f"  REGIONAL n={int(reg.sum())}:  max r={mx[reg].max():.3f}  median={np.median(mx[reg]):.3f}  90th={np.percentile(mx[reg],90):.3f}")
    print(f"  GLOBAL   n={int(glob.sum())}:  max r={mx[glob].max():.3f}  median={np.median(mx[glob]):.3f}")
    ids=np.where(reg)[0][np.argsort(mx[reg])[::-1][:6]]
    for c in ids:
        print(f"   c{c:5d}  r={mx[c]:.2f}  {RL[int(np.argmax(C[:,c]))]:14} "+" ".join(f"{C[j,c]:+.2f}" for j in range(4)))
