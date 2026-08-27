"""Region-regime AR census: for each latent, compare its region-mean activation on AR-regime
timesteps (coverage>=HI) vs clear timesteps (coverage<=LO). AUC>0.5 = fires more during AR regime.
Descriptive. Uses cached tracks + regional_coverage. No re-encode."""
import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
HI=float(os.environ.get("HI","0.9")); LO=float(os.environ.get("LO","0.0"))
q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet")
def auc_cols(A,y,cs=512):
    n1=int(y.sum()); n2=len(y)-n1; C=A.shape[1]; out=np.full(C,0.5)
    if n1==0 or n2==0: return out
    pos=y.astype(bool)
    for i in range(0,C,cs):
        R=np.argsort(np.argsort(A[:,i:i+cs].astype(np.float64),0),0)+1.0
        out[i:i+cs]=(R[pos].sum(0)-n1*(n1+1)/2)/(n1*n2)
    return out
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    print(f"\n===== {SAE}  (AR-regime cov>={HI} vs clear cov<={LO}) =====")
    aucs={}
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); A=d["A_mean"].astype(np.float32); ti=d["tindex"].astype(int)
        cov=q[q.region==r].set_index("time_index")["coverage_frac"].reindex(ti).to_numpy()
        ok=np.isfinite(cov); A=A[ok]; cov=cov[ok]
        arreg=cov>=HI; clear=cov<=LO; keep=arreg|clear
        aucs[r]=auc_cols(A[keep], arreg[keep].astype(int))
        print(f"  {r}: nAR-regime={int(arreg.sum())} nClear={int(clear.sum())} maxAUC={aucs[r].max():.3f}")
    for T in [0.6,0.7,0.8]:
        nreg=np.vstack([aucs[r]>=T for r in REGIONS]).sum(0)
        c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  AUC>={T}: active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    print("  top AR-regime latents per region (AUC>=0.7):")
    for r in REGIONS:
        idx=np.where(aucs[r]>=0.7)[0]; idx=idx[np.argsort(-aucs[r][idx])]
        print(f"    {r} ({len(idx)}): "+", ".join(f"{int(cc)}({aucs[r][int(cc)]:.2f})" for cc in idx[:10]))
