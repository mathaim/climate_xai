import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"
for SAE in ["plain_L8","matry_L8"]:
    try: d=np.load(f"{OUT}/effectsize_{SAE}.npz")
    except FileNotFoundError: print(f"[{SAE}] not ready"); continue
    ma={r:d[f"{r}_mean_ar"] for r in REGIONS}; mn={r:d[f"{r}_mean_no"] for r in REGIONS}
    # floor non-AR mean at 10th pctl of firing concepts -> stable ratio for near-silent concepts
    flo={r:np.percentile(mn[r][mn[r]>0],10) for r in REGIONS}
    print(f"\n##### {SAE}")
    def census(tau):
        act={r:(ma[r]/np.maximum(mn[r],flo[r]))>=tau for r in REGIONS}
        nreg=np.vstack([act[r] for r in REGIONS]).sum(0)
        return {r:(int(act[r].sum()),int((act[r]&(nreg==1)).sum()),int((act[r]&(nreg>=3)).sum())) for r in REGIONS}
    for tau in [1.5,2.0,3.0]:
        print(f"--- ratio>={tau} ---   {'region':20}{'Active':>8}{'Regional':>10}{'Global':>8}")
        c=census(tau)
        for r in REGIONS: print(f"{'':13}{r:20}{c[r][0]:8d}{c[r][1]:10d}{c[r][2]:8d}")
