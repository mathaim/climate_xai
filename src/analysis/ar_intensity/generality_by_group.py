"""Test C: concept generality (# regions active) overall and per Matryoshka nested group (parent/child absorption)."""
import numpy as np, pandas as pd
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.sae_features import load_sae
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"; DEF="magnitude"
def nreg_of(SAE):
    no=np.load(f"{OUT}/nonar_rates_{SAE}.npz")
    F,md=load(SAE,f"region_{DEF}"); reg=md.region.to_numpy()
    magar={r:F[reg==r].mean(0) for r in REGIONS}
    ratio={r:magar[r]/np.maximum(no[f"{r}_{DEF}_no"],1e-3) for r in REGIONS}
    act={r:(ratio[r]>1.1)&(magar[r]>np.median(magar[r][magar[r]>0])) for r in REGIONS}
    return np.vstack([act[r] for r in REGIONS]).sum(0)   # (4096,) # regions each concept active in
def main():
    pre=[int(x) for x in load_sae("matry_L0","cpu")[0].group_sizes]; b=[0]+pre
    groups=[(b[i],b[i+1]) for i in range(len(pre))]
    print("=== Overall: distinct concepts by # regions active ===")
    for SAE in ["plain_L8","matry_L8"]:
        n=nreg_of(SAE)
        print(f"{SAE:9}: active={int((n>=1).sum())} | local(1)={int((n==1).sum())} pair(2)={int((n==2).sum())} "
              f"global(>=3)={int((n>=3).sum())} | mean #regions(active)={n[n>=1].mean():.2f}")
    rows=[]
    for SAE in ["matry_L8","plain_L8"]:
        n=nreg_of(SAE)
        print(f"\n=== {SAE}: generality by group (G0=core ... G4=outer) ===")
        print(f"{'group':>12} {'#active':>8} {'mean #regions':>14} {'%global(>=3)':>13} {'%local(1)':>11}")
        for gi,(lo,hi) in enumerate(groups):
            seg=n[lo:hi]; act=seg[seg>=1]; na=int((seg>=1).sum())
            mg=float(act.mean()) if na else 0.0
            pg=100*float((seg>=3).sum())/max(na,1); pl=100*float((seg==1).sum())/max(na,1)
            print(f"  G{gi}[{lo:>4}:{hi:<4}] {na:>8} {mg:>14.2f} {pg:>12.1f}% {pl:>10.1f}%")
            rows.append(dict(sae=SAE,group=gi,n_active=na,mean_nregions=mg,pct_global=pg,pct_local=pl))
    pd.DataFrame(rows).to_csv("results/ar_intensity/generality_by_group.csv",index=False)
    print("\nSAVED generality_by_group.csv")
if __name__=="__main__": main()
