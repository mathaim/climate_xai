"""Best G0->G4 CONTAINMENT chain (nested firing sets) via widest-path DP maximizing the weakest link."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; B = [0, 256, 512, 1024, 2048, 4096]
def main():
    fire = None
    for r in REGIONS:
        A = np.load(f"{TRACK}/track_matry_{r}.npz")["A_max"].astype(np.float32); f = A > THRESH; del A
        fire = f if fire is None else (fire | f)
    ntime = fire.shape[0]; rate = fire.sum(0).astype(float); ff = fire.astype(np.float32)
    cooc = ff.T @ ff                                   # cooc[p,c] = |fire_p & fire_c|
    C = cooc / np.maximum(rate[None, :], 1)            # C[p,c] = P(parent p | child c)
    UNIV, MINF = 0.5 * ntime, 200
    best = np.full(4096, -1.0); par = np.full(4096, -1, int)
    for g0 in range(B[0], B[1]):
        if MINF < rate[g0] < UNIV: best[g0] = 1.0      # G0 roots: broad but not universal
    for k in range(1, 5):
        pr = np.arange(B[k - 1], B[k])
        for c in range(B[k], B[k + 1]):
            if rate[c] < MINF: continue
            v = (best[pr] >= 0) & (rate[pr] > rate[c])  # parent must be broader
            if not v.any(): continue
            sc = np.minimum(best[pr][v], C[pr[v], c]); j = int(np.argmax(sc))
            best[c] = sc[j]; par[c] = pr[v][j]
    g4 = np.arange(B[4], B[5]); cand = g4[best[g4] >= 0]
    if len(cand) == 0:
        print("NO complete G0->G4 containment chain exists (under rate/broadness constraints)."); return
    leaf = cand[int(np.argmax(best[cand]))]; chain = [leaf]
    x = leaf
    while par[x] >= 0: x = par[x]; chain.append(x)
    chain = chain[::-1]
    print(f"BEST G0->G4 chain: weakest containment link = {best[leaf]:.2f}\n")
    print(f"{'grp':>3} {'concept':>7} {'rate%':>6} {'P(parent|this)':>15}")
    for i, cc in enumerate(chain):
        cont = C[par[cc], cc] if i > 0 else 1.0
        print(f"G{i:>2} {cc:>7} {rate[cc]/ntime*100:>5.0f}% {cont:>15.2f}")
if __name__ == "__main__":
    main()
