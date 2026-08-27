"""Do Matryoshka's cross-regional (global) AR concepts concentrate in the nested core?
Split active concepts (floored ratio>=tau) by generality nreg, look at where they sit across the
5 nested shells. Plain = flat control (index order arbitrary -> expect ~uniform)."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"
BINS=[256,512,1024,2048]; SHELL_SZ=np.array([256,256,512,1024,2048]); NC=4096
idx=np.arange(NC); shell=np.digitize(idx,BINS)  # 0..4
def load(SAE):
    d=np.load(f"{OUT}/effectsize_{SAE}.npz")
    ma={r:d[f"{r}_mean_ar"] for r in REGIONS}; mn={r:d[f"{r}_mean_no"] for r in REGIONS}
    flo={r:np.percentile(mn[r][mn[r]>0],10) for r in REGIONS}
    return np.vstack([(ma[r]/np.maximum(mn[r],flo[r])) for r in REGIONS])  # (4, NC) ratios
def analyze(SAE,tau):
    ratio=load(SAE); act=ratio>=tau; nreg=act.sum(0)
    print(f"\n### {SAE}  tau={tau}   active(nreg>=1)={int((nreg>=1).sum())}   core is 6.25% of latents")
    for label,mask in [("regional nreg==1",nreg==1),("mid nreg==2",nreg==2),("global nreg>=3",nreg>=3)]:
        n=int(mask.sum())
        if n==0: print(f"  {label:16} n=  0"); continue
        cnt=np.array([int((mask&(shell==s)).sum()) for s in range(5)])
        print(f"  {label:16} n={n:4d}  shells(core..out) {cnt.tolist()}  core%={100*cnt[0]/n:4.1f}  med_idx={int(np.median(idx[mask]))}")
    am=nreg>=1
    if int(am.sum())>10:
        c=np.corrcoef(nreg[am],idx[am])[0,1]
        print(f"  corr(nreg, index) over active = {c:+.3f}  (negative => more-general concepts sit more core)")
for SAE in ["matry_L8","plain_L8"]:
    for tau in [1.5,2.0]: analyze(SAE,tau)
