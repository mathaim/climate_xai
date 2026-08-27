import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; shell=np.digitize(np.arange(4096),[256,512,1024,2048])
def pcols(A,y,cs=1024):
    yc=y-y.mean(); yn=np.sqrt((yc*yc).sum()); out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        B=A[:,i:i+cs].astype(np.float64); B=B-B.mean(0)
        out[i:i+cs]=(B*yc[:,None]).sum(0)/(np.sqrt((B*B).sum(0))*yn+1e-12)
    return out
for SAE,stem in [("matry","track_matry"),("plain","track_pool")]:
    freq=np.zeros(4096); nt=0; C=[]
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); ivt=d["ivt"].astype(float); ok=np.isfinite(ivt)
        freq+=(d["A_max"]>0).sum(0); nt+=d["A_max"].shape[0]
        C.append(pcols(d["A_mean"][ok],ivt[ok])); del d
    freq/=nt; C=np.vstack(C); nreg=(C>0.3).sum(0)
    print(f"\n### {SAE}")
    print(f"  ring   n  freq_all  freq_AR   #AR #reg #glob  freq_reg freq_glob")
    for s in range(5):
        m=shell==s; ar=m&(nreg>=1); reg=m&(nreg==1); gl=m&(nreg>=3)
        f=lambda x: freq[x].mean() if x.sum() else 0.0
        print(f"  {s:4d} {int(m.sum()):5d}   {freq[m].mean():.3f}    {f(ar):.3f}   {int(ar.sum()):3d} {int(reg.sum()):3d} {int(gl.sum()):4d}   {f(reg):.3f}    {f(gl):.3f}")
