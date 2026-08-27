"""Emit the per-region AR-latent census as LaTeX (both SAEs), from saved census_mi npz.
Row = region; columns = how many of that region's AR-responsive latents are regional/bi/tri/global.
Unique row = distinct latents per category. No job needed."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
RL=list(REGIONS)
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe",
      "W_S_America":"Western South America","E_Australia":"Eastern Australia"}
META={"plain_L8":("Standard","tab:standardcensus"),"matry_L8":("Matryoshka","tab:matrycensus")}
for SAE in ["plain_L8","matry_L8"]:
    d=np.load(f"/scratch/euh7ys/climate_xai/concept_ivt/census_mi_{SAE}.npz", allow_pickle=True)
    sig=d["sig"]; nreg=sig.sum(0); arch,lab=META[SAE]
    print("\n"+"%"*60)
    print("\\begin{table}\n \\centering\n \\begin{tabular}{l c c c c c}\n \\toprule")
    print("  Region & Regional & Bi-regional & Tri-regional & Global & \\bf{Total} \\\\\n \\midrule")
    for i,r in enumerate(RL):
        c={k:int(((nreg==k)&sig[i]).sum()) for k in (1,2,3,4)}; tot=sum(c.values())
        print(f"   {NAME[r]} & {c[1]} & {c[2]} & {c[3]} & {c[4]} & \\bf{{{tot}}}\\\\")
    u={k:int((nreg==k).sum()) for k in (1,2,3,4)}
    print(" \\midrule")
    print(f"   Unique latents & {u[1]} & {u[2]} & {u[3]} & {u[4]} & \\bf{{{sum(u.values())}}} \\\\")
    print(" \\bottomrule\n \\end{tabular}")
    print(f" \\caption{{AR latent census per region ({arch} SAE, layer 8). A latent is AR-responsive "
          f"in a region if its activation carries statistically significant mutual information with "
          f"node-level AR presence (Higgins detection; label-permutation null, family-wise controlled "
          f"at 1\\%) in the AR-excited direction ($p_{{\\rm AR}}>p_{{\\rm no}}$). Regional = AR-responsive "
          f"in one region, bi-/tri-regional in two/three, global in all four; ``regional'' is relative to "
          f"these four sampled regions. The final row tallies distinct latents.}}")
    print(f" \\label{{{lab}}}\n\\end{{table}}")
