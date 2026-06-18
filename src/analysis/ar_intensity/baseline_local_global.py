"""Baseline (L8 plain, magnitude, no intensity filter): local vs global concepts per region."""
import numpy as np, pandas as pd
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"; SAE="plain_L8"; DEF="magnitude"
def main():
    no=np.load(f"{OUT}/nonar_rates_{SAE}.npz")
    F,md=load(SAE,f"region_{DEF}"); reg=md.region.to_numpy()
    magar={r:F[reg==r].mean(0) for r in REGIONS}
    ratio={r:magar[r]/np.maximum(no[f"{r}_{DEF}_no"],1e-3) for r in REGIONS}
    def floor(r,mode):
        v=magar[r][magar[r]>0]
        return {"median":np.median(v),"p75":np.percentile(v,75),"none":0.0}[mode]
    allrows=[]
    for mode in ["median","p75","none"]:
        act={r:(ratio[r]>1.1)&(magar[r]>floor(r,mode)) for r in REGIONS}
        nreg=np.vstack([act[r] for r in REGIONS]).sum(0)
        print(f"\n=== floor={mode} ===")
        print(f"{'region':13} {'active':>7} {'local':>7} {'global':>7}")
        for r in REGIONS:
            inr=act[r]; loc=int((inr&(nreg==1)).sum()); glb=int((inr&(nreg>=3)).sum())
            print(f"{r:13} {int(inr.sum()):7d} {loc:7d} {glb:7d}")
            allrows.append(dict(floor=mode,region=r,active=int(inr.sum()),local=loc,n_global=glb))
    pd.DataFrame(allrows).to_csv(f"{OUT}/baseline_local_global.csv",index=False)
    print("\nsaved",f"{OUT}/baseline_local_global.csv")
if __name__=="__main__": main()
