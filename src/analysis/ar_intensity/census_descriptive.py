"""Descriptive AR census: per latent, firing-rate gap dP = P(active|AR node) - P(active|non-AR node).
AR-excited = dP>=floor (fires much more on AR); AR-suppressed = dP<=-floor. No significance claim."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
RL=list(REGIONS); FLOORS=[0.01,0.03,0.05,0.10]
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe","W_S_America":"Western South America","E_Australia":"Eastern Australia"}
for SAE in ["plain_L8","matry_L8"]:
    d=np.load(f"/scratch/euh7ys/climate_xai/concept_ivt/census_fire_{SAE}.npz",allow_pickle=True)
    dP=d["dP"]  # (4,4096)
    print(f"\n===== {SAE} =====")
    for fl in FLOORS:
        ne=(dP>=fl).sum(0); ns=(dP<=-fl).sum(0)
        c={k:int((ne==k).sum()) for k in (1,2,3,4)}
        print(f"  |dP|>={fl}:  AR-excited active={int((ne>=1).sum())} (reg {c[1]} bi {c[2]} tri {c[3]} glob {c[4]})   AR-suppressed active={int((ns>=1).sum())}")
    fl=0.01; ne=(dP>=fl).sum(0)
    print(f"\n  --- AR-excited latents (dP>={fl}, fire >= {int(fl*100)} pts more on AR), by region, top by dP ---")
    for i,r in enumerate(RL):
        idx=np.where(dP[i]>=fl)[0]; idx=idx[np.argsort(-dP[i][idx])]
        print(f"    {NAME[r]} ({len(idx)}): "+", ".join(f"{int(c)}({dP[i][int(c)]:+.2f})" for c in idx[:10]))
    # census table (AR-excited at 0.05)
    print("\n  %--- table (AR-excited, dP>=0.05) ---")
    print("   Region & Regional & Bi & Tri & Global & Total \\\\")
    for i,r in enumerate(RL):
        cc={k:int(((ne==k)&(dP[i]>=fl)).sum()) for k in (1,2,3,4)}
        print(f"   {NAME[r]} & {cc[1]} & {cc[2]} & {cc[3]} & {cc[4]} & {sum(cc.values())} \\\\")
    u={k:int((ne==k).sum()) for k in (1,2,3,4)}
    print(f"   Unique & {u[1]} & {u[2]} & {u[3]} & {u[4]} & {sum(u.values())} \\\\")
