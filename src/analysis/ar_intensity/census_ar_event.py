import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
COV=float(os.environ.get("COV","0.75"))
q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet")
def auc(comb,n1,cs=512):
    NC=comb.shape[1]; n2=comb.shape[0]-n1; out=np.zeros(NC)
    for i in range(0,NC,cs):
        R=np.argsort(np.argsort(comb[:,i:i+cs].astype(np.float64),0),0)+1.0
        out[i:i+cs]=(R[:n1].sum(0)-n1*(n1+1)/2)/max(n1*n2,1)
    return out
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    C=[]; nar=[]
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); A=d["A_mean"]; ti=d["tindex"].astype(int)
        cov=q[q.region==r].set_index("time_index")["coverage_frac"].reindex(ti).to_numpy()
        ar=cov>=COV; no=cov==0; nar.append(int(ar.sum()))
        C.append(auc(np.vstack([A[ar],A[no]]),int(ar.sum()))); del d
    C=np.vstack(C)
    print(f"\n##### {SAE}  (AR-event AUC, coverage>={COV})  max AUC={C.max():.3f}  n_AR/region={nar}")
    for tau in [0.55,0.60,0.65,0.70]:
        nreg=(C>tau).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  AUC>{tau}: active={int((nreg>=1).sum())}  reg={c[1]} bi={c[2]} tri={c[3]} global={c[4]}")
