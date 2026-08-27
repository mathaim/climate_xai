"""Existence census (region-aggregate, circular-shift null). Per latent, per region:
corr(A_mean, region IVT) over the full record. Null: circular-shift IVT by random offsets >= 1 year
(events de-align, autocorrelation preserved), NSHIFT times; family-wise ceiling = max over shifts of
the max-over-4096 surrogate corr. A latent SURVIVES in a region if its true corr exceeds that ceiling
(FWER ~ 1/NSHIFT). Existence claim + regional/bi/tri/global over survivors. Cached tracks."""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
D="/scratch/euh7ys/climate_xai/concept_ivt"; NSHIFT=int(os.environ.get("NSHIFT","200")); YR=1461
for SAE,stem in [("plain_L8","track_pool"),("matry_L8","track_matry")]:
    print(f"\n===== {SAE} =====")
    surv={}; rng=np.random.default_rng(0)
    for r in REGIONS:
        d=np.load(f"{D}/{stem}_{r}.npz"); A=d["A_mean"].astype(np.float64); iv=d["ivt"].astype(float)
        ok=np.isfinite(iv); A=A[ok]; iv=iv[ok]; T=len(iv)
        Az=A-A.mean(0); nA=np.sqrt((Az*Az).sum(0))+1e-12
        def corr(y): yz=y-y.mean(); return (Az.T@yz)/(nA*np.sqrt((yz*yz).sum())+1e-12)
        obs=corr(iv)
        nmax=[corr(np.roll(iv,int(rng.integers(YR,T-YR)))).max() for _ in range(NSHIFT)]
        ceil=max(nmax); s=obs>ceil; surv[r]=s
        print(f"  {r}: maxCorr={obs.max():.2f} shift-ceiling={ceil:.2f} #survive={int(s.sum())}")
    M=np.vstack([surv[r] for r in REGIONS]); nreg=M.sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
    print(f"  CENSUS (survive circular-shift null): active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    print(f"  global: {np.where(nreg==4)[0].tolist()}")
