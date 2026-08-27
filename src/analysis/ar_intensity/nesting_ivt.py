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
rank=lambda x:np.argsort(np.argsort(x)).astype(float); pc=lambda a,b:np.corrcoef(a,b)[0,1]
for SAE,stem in [("matry_L8","track_matry"),("plain_L8","track_pool")]:
    C=[]
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); ivt=d["ivt"].astype(float); ok=np.isfinite(ivt)
        C.append(spearman_cols(d["A_max"][ok],ivt[ok])); del d
    C=np.vstack(C); G=C.min(0); Gmean=C.mean(0); Gmax=C.max(0); resp=Gmax>0.3
    print(f"\n### {SAE}  responsive(max-corr>0.3)={int(resp.sum())}")
    print(" per-region median corr:",[round(float(np.median(C[i])),3) for i in range(4)])
    print(" generality min-corr: #>0.2:%d #>0.3:%d #>0.4:%d"%((G>0.2).sum(),(G>0.3).sum(),(G>0.4).sum()))
    print(" corr(min,index)=%+.3f  corr(mean,index)=%+.3f  corr(max,index)=%+.3f"%(
          pc(rank(G[resp]),rank(idx[resp])),pc(rank(Gmean[resp]),rank(idx[resp])),pc(rank(Gmax[resp]),rank(idx[resp]))))
    glob=resp&(G>=np.quantile(G[resp],0.9)); loc=resp&(G<=np.quantile(G[resp],0.5))
    for lab,mask in [("GLOBAL top-decile min",glob),("LOCAL bottom-half min",loc)]:
        n=int(mask.sum()); cnt=[int((mask&(shell==s)).sum()) for s in range(5)]
        print(f"  {lab}: n={n} shells{cnt} core%={100*cnt[0]/max(n,1):.1f} med_idx={int(np.median(idx[mask])) if n else 0}")
