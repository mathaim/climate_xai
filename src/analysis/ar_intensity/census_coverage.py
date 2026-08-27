import numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); rng=np.random.default_rng(0)
def scols(A,y,cs=1024):
    ry=np.argsort(np.argsort(y)).astype(np.float64); ry-=ry.mean(); ryn=np.sqrt((ry*ry).sum())
    out=np.zeros(A.shape[1])
    for i in range(0,A.shape[1],cs):
        R=np.argsort(np.argsort(A[:,i:i+cs].astype(np.float64),0),0).astype(np.float64); R-=R.mean(0)
        out[i:i+cs]=(R*ry[:,None]).sum(0)/(np.sqrt((R*R).sum(0))*ryn+1e-12)
    return out
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    C=[]; nullmax=[]
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); A=d["A_mean"]; ti=d["tindex"].astype(int)
        cov=q[q.region==r].set_index("time_index")["coverage_frac"].reindex(ti).to_numpy()
        ok=np.isfinite(cov); A=A[ok]; cov=cov[ok]
        C.append(scols(A,cov))
        for _ in range(50):
            off=int(rng.integers(500,len(cov)-500)); nullmax.append(np.abs(scols(A,np.roll(cov,off))).max())
        del d
    C=np.vstack(C); nc=np.percentile(nullmax,99)
    print(f"\n##### {SAE}  corr(A_mean, coverage)  max r={C.max():.3f}  99th-pct chance r={nc:.3f}")
    for tau in [round(nc,2),0.3,0.4]:
        nreg=(C>tau).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  r>{tau}: active={int((nreg>=1).sum())}  reg={c[1]} bi={c[2]} tri={c[3]} global={c[4]}")
