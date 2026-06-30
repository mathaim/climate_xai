"""Find child->parent pairs where the child fires ONLY when the parent fires: P(parent|child) ~ 1,
parent broader than child, parent NOT near-universal (excludes the always-on core like 99)."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; REGN = list(REGIONS)
def main():
    fire = None; regcnt = np.zeros((4, 4096))
    for ri, r in enumerate(REGN):
        A = np.load(f"{TRACK}/track_matry_{r}.npz")["A_max"].astype(np.float32)
        f = A > THRESH; del A
        fire = f if fire is None else (fire | f); regcnt[ri] = f.sum(0)
    ntime = fire.shape[0]; rate = fire.sum(0).astype(np.float64)
    domreg = np.array(REGN)[regcnt.argmax(0)]; ids = np.arange(4096)
    ff = fire.astype(np.float32); cooc = ff.T @ ff           # cooc[p,c] = |fire_p & fire_c|
    UNIV = 0.5 * ntime; MINFIRE = 200
    res = []
    for c in range(4096):
        if rate[c] < MINFIRE: continue
        cont = cooc[:, c] / rate[c]                          # P(p | c) for every p
        mask = (ids != c) & (rate > rate[c]) & (rate < UNIV)  # parent broader, not universal
        cm = np.where(mask, cont, -1.0); p = int(np.argmax(cm))
        if cm[p] > 0.90:
            res.append((c, p, cm[p], int(rate[c]), int(rate[p])))
    res.sort(key=lambda x: -x[2])
    print(f"{len(res)} child->parent pairs with P(parent|child) > 0.90 and a non-universal parent\n")
    print(f"{'child':>6} {'child_reg':>10} {'parent':>7} {'parent_reg':>11} {'P(p|c)':>7} {'c_fires':>8} {'p_fires':>8} {'p_rate%':>8}")
    for c, p, ct, rc, rp in res[:50]:
        print(f"{c:>6} {domreg[c][:9]:>10} {p:>7} {domreg[p][:9]:>11} {ct:>7.2f} {rc:>8} {rp:>8} {100*rp/ntime:>7.0f}%")
if __name__ == "__main__":
    main()
