import numpy as np
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"; DEF="magnitude"; SAE="plain_L8"
no=np.load(f"{OUT}/nonar_rates_{SAE}.npz")
F,md=load(SAE,f"region_{DEF}"); reg=md.region.to_numpy()
magar={r:F[reg==r].mean(0) for r in REGIONS}
sdar ={r:F[reg==r].std(0)  for r in REGIONS}
nom  ={r:no[f"{r}_{DEF}_no"] for r in REGIONS}
def counts(act):
    nreg=np.vstack([act[r] for r in REGIONS]).sum(0)
    return {r:(int(act[r].sum()),int((act[r]&(nreg==1)).sum()),int((act[r]&(nreg>=3)).sum())) for r in REGIONS}
def show(tag,act):
    print(f"\n=== {tag} ==="); print(f"{'region':24}{'Active':>8}{'Regional':>10}{'Global':>8}")
    c=counts(act)
    for r in REGIONS: print(f"{r:24}{c[r][0]:8d}{c[r][1]:10d}{c[r][2]:8d}")
show("ratio>1.1 + median floor (current)",
     {r:(magar[r]/np.maximum(nom[r],1e-3)>1.1)&(magar[r]>np.median(magar[r][magar[r]>0])) for r in REGIONS})
for k in [1.0,1.5,2.0]:
    show(f"(mean_AR - mean_nonAR)/SD_AR >= {k}  [AR-time SD proxy]",
         {r:((magar[r]-nom[r])/np.maximum(sdar[r],1e-6))>=k for r in REGIONS})
