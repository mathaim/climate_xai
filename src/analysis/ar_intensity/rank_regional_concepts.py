"""Top LOCAL concepts per region by activation (mag_ar), L8 plain magnitude baseline."""
import numpy as np, pandas as pd
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"; SAE="plain_L8"; DEF="magnitude"; TOPN=10
def main():
    no=np.load(f"{OUT}/nonar_rates_{SAE}.npz")
    F,md=load(SAE,f"region_{DEF}"); reg=md.region.to_numpy()
    magar={r:F[reg==r].mean(0) for r in REGIONS}
    ratio={r:magar[r]/np.maximum(no[f"{r}_{DEF}_no"],1e-3) for r in REGIONS}
    act={}
    for r in REGIONS:
        fl=np.median(magar[r][magar[r]>0]); act[r]=(ratio[r]>1.1)&(magar[r]>fl)
    nreg=np.vstack([act[r] for r in REGIONS]).sum(0)
    rows=[]
    for r in REGIONS:
        idx=np.where(act[r]&(nreg==1))[0]
        idx=idx[np.argsort(-magar[r][idx])][:TOPN]
        print(f"\n=== {r}: top {TOPN} LOCAL concepts by mag_ar ===")
        print(f"{'concept':>8} {'mag_ar':>9} {'ratio':>8}")
        for c in idx:
            print(f"{int(c):>8d} {magar[r][c]:9.3f} {ratio[r][c]:8.2f}")
            rows.append(dict(region=r,concept=int(c),mag_ar=float(magar[r][c]),ratio=float(ratio[r][c])))
    pd.DataFrame(rows).to_csv(f"{OUT}/top_local_concepts.csv",index=False)
    print("\nsaved",f"{OUT}/top_local_concepts.csv")
if __name__=="__main__": main()
