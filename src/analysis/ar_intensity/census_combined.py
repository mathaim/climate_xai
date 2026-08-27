"""Combined AR-concept census (region-aggregate). A latent is a true AR concept in a region if:
 (a) PRESENCE: it activates more during AR than non-AR timesteps (mean A_mean over AR events
     [coverage>=0.5] > RATIO x mean over clear events [coverage==0]) -- uses the non-AR baseline;
 (b) INTENSITY: its regional firing scales with AR intensity (ar_corr > TAU) -- AR-specific,
     so general-storm latents (no intensity scaling) drop out.
Both required. Global concepts (e.g. 99) pass both in all 4 regions. Cached tracks+coverage+ar_corr."""
import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
D="/scratch/euh7ys/climate_xai/concept_ivt"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
TAU=float(os.environ.get("TAU","0.3")); RATIO=float(os.environ.get("RATIO","1.5"))
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe","W_S_America":"Western South America","E_Australia":"Eastern Australia"}
q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); RL=list(REGIONS)
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    print(f"\n===== {SAE} =====")
    corr={r:np.load(f"{D}/ar_corr_{SAE}_{r}.npy") for r in RL}; spec={}
    for r in RL:
        d=np.load(f"{D}/{stem}_{r}.npz"); A=d["A_mean"].astype(np.float64); ti=d["tindex"].astype(int)
        cov=q[q.region==r].set_index("time_index")["coverage_frac"].reindex(ti).to_numpy()
        ok=np.isfinite(cov); A=A[ok]; cov=cov[ok]
        pAR=A[cov>=0.5].mean(0); pno=A[cov==0].mean(0)+1e-9; spec[r]=pAR/pno
        print(f"  {r}: nAR={int((cov>=0.5).sum())} nClear={int((cov==0).sum())} maxCorr={corr[r].max():.2f} medSpec={np.median(spec[r]):.2f}")
    def comb(T,R): return {r:(corr[r]>T)&(spec[r]>R) for r in RL}
    for T,R in [(TAU,RATIO),(0.3,2.0),(0.4,1.5)]:
        M=comb(T,R); nreg=np.vstack([M[r] for r in RL]).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  corr>{T} & spec>{R}: active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    M=comb(TAU,RATIO); nreg=np.vstack([M[r] for r in RL]).sum(0)
    print(f"  GLOBAL (corr>{TAU} & spec>{RATIO} all 4): {np.where(nreg==4)[0].tolist()}")
    print(f"  %--- table (corr>{TAU} & spec>{RATIO}) ---")
    print("   Region & Regional & Bi-regional & Tri-regional & Global & \\bf{Total} \\\\")
    for r in RL:
        cc={k:int(((nreg==k)&M[r]).sum()) for k in (1,2,3,4)}
        print(f"   {NAME[r]} & {cc[1]} & {cc[2]} & {cc[3]} & {cc[4]} & \\bf{{{sum(cc.values())}}}\\\\")
    u={k:int((nreg==k).sum()) for k in (1,2,3,4)}
    print(f"   Unique latents & {u[1]} & {u[2]} & {u[3]} & {u[4]} & \\bf{{{sum(u.values())}}} \\\\")
    np.savez(f"{D}/census_combined_{SAE}.npz",corr=np.vstack([corr[r] for r in RL]),spec=np.vstack([spec[r] for r in RL]),regions=np.array(RL))
