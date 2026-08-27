"""Recount the node-level AR census at several effect-size floors, from the SAVED census_fire npz
(no re-run). Criterion: significant (FWER) AND dP >= floor. Prints reg/bi/tri/global per floor and
the regional latent IDs at a chosen floor. 'regional/global' relative to the 4 sampled regions."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
RL=list(REGIONS); BARS=[0.02,0.03,0.04,0.05]; SHOW=0.03
for SAE in ["plain_L8","matry_L8"]:
    d=np.load(f"/scratch/euh7ys/climate_xai/concept_ivt/census_fire_{SAE}.npz", allow_pickle=True)
    dP=d["dP"]; sig=d["sig"]                      # (4,4096)
    print(f"\n===== {SAE} =====")
    for bar in BARS:
        M=sig&(dP>=bar); nreg=M.sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  sig & dP>={bar:.2f}: active={int((nreg>=1).sum())} regional={c[1]} bi={c[2]} tri={c[3]} global={c[4]}")
    M=sig&(dP>=SHOW); nreg=M.sum(0)
    print(f"  --- REGIONAL latents (sig & dP>={SHOW}, in ONLY 1 region), top by dP ---")
    for i,r in enumerate(RL):
        ro=np.where((nreg==1)&M[i])[0]; ro=ro[np.argsort(-dP[i][ro])]
        print(f"    {r} ({len(ro)}): "+", ".join(f"{int(x)}({dP[i][int(x)]:.2f})" for x in ro[:15]))
