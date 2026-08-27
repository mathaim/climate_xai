"""Direct-mask dependence census, MAGNITUDE. Region-AR = any AR (coverage>0); no-AR = none.
Per latent per region: AUC of region-mean activation A_mean between region-AR and no-AR timesteps.
Magnitude (not binary) so always-on concepts (99) -- whose activation rises during ARs -- are caught.
Direct mask, full record, confounds allowed (dependence claim)."""
import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
D="/scratch/euh7ys/climate_xai/concept_ivt"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); RL=list(REGIONS)
def auc(A,y,cs=512):
    n1=int(y.sum()); n2=len(y)-n1; C=A.shape[1]; out=np.full(C,0.5); pos=y.astype(bool)
    if n1==0 or n2==0: return out
    for i in range(0,C,cs):
        R=np.argsort(np.argsort(A[:,i:i+cs].astype(np.float64),0),0)+1.0
        out[i:i+cs]=(R[pos].sum(0)-n1*(n1+1)/2)/(n1*n2)
    return out
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    print(f"\n===== {SAE} =====")
    au={}
    for r in RL:
        d=np.load(f"{D}/{stem}_{r}.npz"); A=d["A_mean"].astype(np.float32); ti=d["tindex"].astype(int)
        cov=q[q.region==r].set_index("time_index")["coverage_frac"].reindex(ti).to_numpy()
        ok=np.isfinite(cov); A=A[ok]; AR=cov[ok]>0
        au[r]=auc(A,AR.astype(int)); print(f"  {r}: nAR={int(AR.sum())} nNO={int((~AR).sum())} maxAUC={au[r].max():.3f}")
    for th in (0.6,0.7,0.8):
        nreg=np.vstack([au[r]>th for r in RL]).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  AUC>{th}: active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    g=np.where(np.vstack([au[r]>0.7 for r in RL]).sum(0)==4)[0]; print(f"  global(AUC>0.7 all4): {g.tolist()}")
    np.savez(f"{D}/anyAR_mag_{SAE}.npz",auc=np.vstack([au[r] for r in RL]),regions=np.array(RL))
