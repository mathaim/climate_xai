import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"; BINS=[256,512,1024,2048]; idx=np.arange(4096); shell=np.digitize(idx,BINS)
def aucs(SAE):
    d=np.load(f"{OUT}/auc_{SAE}.npz"); A=np.zeros((4,4096))
    for ri,r in enumerate(REGIONS):
        Ha=d[f"ar_{r}"].astype(np.float64); Hn=d[f"no_{r}"].astype(np.float64)
        Na=Ha.sum(1); Nn=Hn.sum(1); cn=np.cumsum(Hn,1); below=cn-Hn
        A[ri]=((Ha*below).sum(1)+0.5*(Ha*Hn).sum(1))/np.maximum(Na*Nn,1)
    return A, d["Fsum"]/max(int(d["Fn"][0]),1)
rank=lambda x:np.argsort(np.argsort(x)).astype(float)
pc=lambda a,b:np.corrcoef(a,b)[0,1]
for SAE in ["plain_L8","matry_L8"]:
    try: A,F=aucs(SAE)
    except FileNotFoundError: print(f"[{SAE}] not ready"); continue
    G=A.min(0); Gmax=A.max(0); resp=Gmax>0.55
    print(f"\n##### {SAE}   AR-responsive(max-AUC>0.55)={int(resp.sum())}")
    print(" per-region median AUC:",[round(float(np.median(A[i])),3) for i in range(4)])
    print(" per-region max AUC   :",[round(float(np.max(A[i])),3) for i in range(4)])
    print(" generality min-AUC:  #>0.55:%d #>0.6:%d #>0.65:%d"%((G>0.55).sum(),(G>0.6).sum(),(G>0.65).sum()))
    print(" corr(min-AUC generality, index) = %+.3f"%pc(rank(G[resp]),rank(idx[resp])))
    print(" corr(F firing-strength, index)  = %+.3f"%pc(rank(F[resp]),rank(idx[resp])))
    glob=resp&(G>=np.quantile(G[resp],0.9)); loc=resp&(G<=np.quantile(G[resp],0.5))
    for lab,mask in [("GLOBAL top-decile min-AUC",glob),("LOCAL bottom-half min-AUC",loc)]:
        n=int(mask.sum()); cnt=[int((mask&(shell==s)).sum()) for s in range(5)]
        print(f"  {lab}: n={n} shells{cnt} core%={100*cnt[0]/max(n,1):.1f} med_idx={int(np.median(idx[mask])) if n else 0}")
