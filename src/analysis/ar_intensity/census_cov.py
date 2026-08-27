"""Direct-mask GRADIENT census. Per latent per region: Pearson corr(A_mean, coverage_frac) over the
full record -- activation vs how much of the region the AR mask covers (0 = no AR .. 1 = full). Uses
the mask directly, keeps the intensity gradient (not a weak any-vs-none binary), includes non-AR frames.
Confounds allowed (dependence). Catches always-on concepts (99) whose activation scales with AR extent."""
import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
D="/scratch/euh7ys/climate_xai/concept_ivt"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); RL=list(REGIONS)
def pear(A,y):
    Az=A-A.mean(0); yz=y-y.mean(); return (Az.T@yz)/(np.sqrt((Az*Az).sum(0))*np.sqrt((yz*yz).sum())+1e-12)
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    print(f"\n===== {SAE} =====")
    corr={}
    for r in RL:
        d=np.load(f"{D}/{stem}_{r}.npz"); A=d["A_mean"].astype(np.float64); ti=d["tindex"].astype(int)
        cov=q[q.region==r].set_index("time_index")["coverage_frac"].reindex(ti).to_numpy()
        ok=np.isfinite(cov); A=A[ok]; cov=cov[ok]
        corr[r]=pear(A,cov); print(f"  {r}: n={len(cov)} maxCorr={corr[r].max():.2f}")
    for th in (0.2,0.3,0.4):
        nreg=np.vstack([corr[r]>th for r in RL]).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  corr>{th}: active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    g=np.where(np.vstack([corr[r]>0.3 for r in RL]).sum(0)==4)[0]; print(f"  global(corr>0.3 all4): {g.tolist()}")
    np.savez(f"{D}/census_cov_{SAE}.npz",corr=np.vstack([corr[r] for r in RL]),regions=np.array(RL))
