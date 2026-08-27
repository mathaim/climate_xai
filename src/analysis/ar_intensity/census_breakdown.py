import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
T="/scratch/euh7ys/climate_xai/concept_ivt"
def pcols(A,y,cs=1024):
    yc=y-y.mean(); yn=np.sqrt((yc*yc).sum()); out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        B=A[:,i:i+cs].astype(np.float64); B=B-B.mean(0)
        out[i:i+cs]=(B*yc[:,None]).sum(0)/(np.sqrt((B*B).sum(0))*yn+1e-12)
    return out
for S,st in [("plain","track_pool"),("matry","track_matry")]:
    C=[]
    for r in REGIONS:
        d=np.load(f"{T}/{st}_{r}.npz"); iv=d["ivt"].astype(float); ok=np.isfinite(iv)
        C.append(pcols(d["A_mean"][ok],iv[ok])); del d
    C=np.vstack(C); act=C>0.3; nreg=act.sum(0)
    print(f"\n### {S}   region: Active Reg(1) Bi(2) Tri(3) Global(4)")
    for i,r in enumerate(REGIONS):
        a=act[i]; row=[int(a.sum())]+[int((a&(nreg==k)).sum()) for k in (1,2,3,4)]
        print(f"  {r:16} "+" ".join(f"{x:4d}" for x in row))
    print("  unique concepts:", {f"n={k}":int((nreg==k).sum()) for k in (1,2,3,4)})
