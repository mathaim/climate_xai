"""Re-select a parent's children by INTENSITY TUNING SLOPE (activation rising with IVT in
one region, flat elsewhere), not by |r|. Combines co-firing dependence + per-region rise."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"
COF = "/scratch/euh7ys/climate_xai/cofire/cofire_matry_L8.npz"
PARENT = 99
def main():
    d = np.load(COF); C = d["cofire"].astype(float); f = d["fire"].astype(float)
    rise = np.zeros((4, 4096))   # per-region: mean activation(intense) - mean(typical)
    for ri, r in enumerate(REGIONS):
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); A = t["A_max"]; ivt = t["ivt"].astype(float)
        ok = np.isfinite(ivt); A = A[ok]; ivt = ivt[ok]
        hi = ivt >= np.quantile(ivt, 0.90); lo = ivt <= np.quantile(ivt, 0.50)
        rise[ri] = A[hi].mean(0) - A[lo].mean(0); del t, A
    outer = np.arange(2048, 4096); Ppc = C[PARENT, outer] / np.maximum(f[outer], 1)
    cand = []
    for j, c in enumerate(outer):
        if Ppc[j] < 0.5 or f[c] < 50: continue
        rr = rise[:, c]; b = int(rr.argmax())
        if rr[b] < 0.02: continue                      # must actually rise somewhere
        spec = rr[b] - np.sort(rr)[-2]                 # one-region dominance
        cand.append((c, list(REGIONS)[b], round(float(rr[b]), 3), round(float(spec), 3),
                     round(float(Ppc[j]), 2)))
    cand.sort(key=lambda x: -(x[3] * x[4]))
    print(f"PARENT {PARENT} children re-selected by rising-intensity tuning:")
    print(f"{'child':>6}{'rise_region':>14}{'rise':>7}{'specificity':>12}{'P(par|child)':>13}")
    for c, reg, ri_, sp, pp in cand[:15]:
        print(f"{c:>6}{reg:>14}{ri_:>7}{sp:>12}{pp:>13}")
    print("DONE", len(cand), "rising children")
if __name__ == "__main__":
    main()
