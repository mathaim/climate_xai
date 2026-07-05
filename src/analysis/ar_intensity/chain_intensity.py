"""Best G0->G4 chain where each child fires for MORE INTENSE ARs (median IVT-at-firing increases),
with containment as the link quality. Tests whether an intensity-refinement ladder nests cleanly."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; B = [0, 256, 512, 1024, 2048, 4096]
def main():
    fire = None; IVT = None
    for r in REGIONS:
        d = np.load(f"{TRACK}/track_matry_{r}.npz"); A = d["A_max"].astype(np.float32); iv = d["ivt"].astype(float)
        f = A > THRESH; del A
        fire = f if fire is None else (fire | f); IVT = iv if IVT is None else np.maximum(IVT, iv)
    ntime = fire.shape[0]; rate = fire.sum(0).astype(float); ff = fire.astype(np.float32)
    C = (ff.T @ ff) / np.maximum(rate[None, :], 1)          # C[p,c] = P(parent p | child c)
    ok = np.isfinite(IVT); medIVT = np.full(4096, np.nan)
    for c in range(4096):
        m = fire[:, c] & ok
        if m.sum() >= 50: medIVT[c] = np.median(IVT[m])     # typical IVT when this concept fires
    UNIV, MINF = 0.5 * ntime, 200
    best = np.full(4096, -1.0); par = np.full(4096, -1, int)
    for g0 in range(B[0], B[1]):
        if MINF < rate[g0] < UNIV and np.isfinite(medIVT[g0]): best[g0] = 1.0
    for k in range(1, 5):
        pr = np.arange(B[k - 1], B[k])
        for c in range(B[k], B[k + 1]):
            if rate[c] < MINF or not np.isfinite(medIVT[c]): continue
            v = (best[pr] >= 0) & (rate[pr] > rate[c]) & (medIVT[c] > medIVT[pr])   # child rarer AND more intense
            if not v.any(): continue
            sc = np.minimum(best[pr][v], C[pr[v], c]); j = int(np.argmax(sc))
            best[c] = sc[j]; par[c] = pr[v][j]
    g4 = np.arange(B[4], B[5]); cand = g4[best[g4] >= 0]
    if len(cand) == 0:
        print("NO intensity-increasing G0->G4 chain exists."); return
    leaf = cand[int(np.argmax(best[cand]))]; chain = [leaf]; x = leaf
    while par[x] >= 0: x = par[x]; chain.append(x)
    chain = chain[::-1]
    print(f"BEST intensity-graded chain: weakest containment link = {best[leaf]:.2f}\n")
    print(f"{'grp':>3} {'concept':>7} {'rate%':>6} {'medIVT@fire':>12} {'P(parent|this)':>15}")
    for i, cc in enumerate(chain):
        cont = C[par[cc], cc] if i > 0 else 1.0
        print(f"G{i:>2} {cc:>7} {rate[cc]/ntime*100:>5.0f}% {medIVT[cc]:>12.0f} {cont:>15.2f}")
if __name__ == "__main__":
    main()
