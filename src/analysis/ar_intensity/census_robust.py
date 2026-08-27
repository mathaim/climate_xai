import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
T="/scratch/euh7ys/climate_xai/concept_ivt"
def pcols(A,y,cs=1024):
    yc=y-y.mean(); yn=np.sqrt((yc*yc).sum()); out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        B=A[:,i:i+cs].astype(np.float64); B=B-B.mean(0)
        out[i:i+cs]=(B*yc[:,None]).sum(0)/(np.sqrt((B*B).sum(0))*yn+1e-12)
    return out
for S,st in [("Standard","track_pool"),("Matryoshka","track_matry")]:
    C=[]
    for r in REGIONS:
        d=np.load(f"{T}/{st}_{r}.npz"); iv=d["ivt"].astype(float); ok=np.isfinite(iv)
        C.append(pcols(d["A_mean"][ok],iv[ok])); del d
    C=np.vstack(C)
    print(f"\n{S}:   tau    Reg  Bi  Tri  Global(all4)  Total   Reg:Global")
    for tau in [0.20,0.25,0.30,0.40,0.50]:
        nreg=(C>tau).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        tot=int((nreg>=1).sum()); ratio=c[1]/max(c[4],1)
        print(f"    {tau:.2f}  {c[1]:5d}{c[2]:4d}{c[3]:5d}{c[4]:9d}    {tot:5d}    {ratio:.0f}:1")
