import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; BINS=[256,512,1024,2048]; idx=np.arange(4096); shell=np.digitize(idx,BINS); RL=list(REGIONS)
def pearson_cols(A,y,cs=1024):
    yc=y-y.mean(); yn=np.sqrt((yc*yc).sum()); out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        B=A[:,i:i+cs].astype(np.float64); B=B-B.mean(0)
        out[i:i+cs]=(B*yc[:,None]).sum(0)/(np.sqrt((B*B).sum(0))*yn+1e-12)
    return out
for SAE,stem in [("matry_L8","track_matry"),("plain_L8","track_pool")]:
    C=[]
    for r in RL:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); ivt=d["ivt"].astype(float); ok=np.isfinite(ivt)
        C.append(pearson_cols(d["A_mean"][ok],ivt[ok])); del d
    C=np.vstack(C)
    for tau in [0.3,0.5]:
        nreg=(C>tau).sum(0)
        print(f"\n##### {SAE}  (A_mean Pearson, r>{tau})   core=shell0=6.25%")
        for lab,m in [("GLOBAL >=3",nreg>=3),("two",nreg==2),("REGIONAL 1",nreg==1)]:
            n=int(m.sum()); cnt=[int((m&(shell==s)).sum()) for s in range(5)]
            print(f"  {lab:11} n={n:3d} shells[core..out]{cnt} core%={100*cnt[0]/max(n,1):.1f} med_idx={int(np.median(idx[m])) if n else 0}")
        print(f"  GLOBAL (id, shell, r: {'/'.join(r[:4] for r in RL)}):")
        for c in np.where(nreg>=3)[0]:
            print(f"    c{c:<5d} s{int(np.digitize(c,BINS))}  "+" ".join(f"{C[j,c]:+.2f}" for j in range(4)))
        print(f"  REGIONAL in CORE (shell 0):")
        for c in np.where((nreg==1)&(idx<256))[0]:
            print(f"    c{c:<5d}  "+" ".join(f"{C[j,c]:+.2f}" for j in range(4))+f"   [{RL[int(np.argmax(C[:,c]))]}]")
