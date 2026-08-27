import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; BINS=[256,512,1024,2048]
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
    C=np.vstack(C); nreg=(C>0.3).sum(0); glob=np.where(nreg>=3)[0]
    print(f"\n##### {SAE}  —  {len(glob)} global concepts (corr>0.3 in >=3 regions)")
    print(f"  {'id':>5} {'shell':>5}   "+"  ".join(f"{r[:6]:>6}" for r in REGIONS)+"   #reg")
    for c in glob:
        print(f"  {c:5d}   s{int(np.digitize(c,BINS))}   "+"  ".join(f"{C[j,c]:+.2f}" for j in range(4))+f"     {int(nreg[c])}")
