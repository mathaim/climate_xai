"""Trace a matryoshka lineage G0->...->G4 backward from a G4 specialist by co-firing lift."""
import numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; AR_START = pd.Timestamp("1979-01-01")
REGN = list(REGIONS); MON = "J F M A M J J A S O N D".split(); BOUNDS = [0, 256, 512, 1024, 2048, 4096]
def group(i):
    for g in range(5):
        if BOUNDS[g] <= i < BOUNDS[g + 1]: return g
def main():
    fire = None; regcnt = np.zeros((4, 4096)); mocnt = np.zeros((12, 4096))
    for ri, r in enumerate(REGN):
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); A = t["A_max"].astype(np.float32); ti = t["tindex"]
        f = A > THRESH; del A
        fire = f.copy() if fire is None else (fire | f)
        regcnt[ri] = f.sum(0)
        mo = (AR_START + pd.to_timedelta(6 * (ti - 1), unit="h")).month.values
        for mm in range(1, 13): mocnt[mm - 1] += f[mo == mm].sum(0)
        del f, t
    ntime = fire.shape[0]; rate = fire.mean(0)
    def conc(c): tot = c.sum(0); fr = c / np.maximum(tot, 1); return 1 - (-(fr*np.log(fr+1e-12)).sum(0)/np.log(c.shape[0]))
    rc, mc = conc(regcnt), conc(mocnt); domreg = np.argmax(regcnt, 0); dommo = np.argmax(mocnt, 0)
    def fp(c): return f"G{group(c)}  rate={rate[c]*100:4.1f}%  spec reg/mon={rc[c]:.2f}/{mc[c]:.2f}  {REGN[domreg[c]][:7]:>7}/{MON[dommo[c]]}"
    def trace(leaf):
        cur = leaf; chain = [leaf]
        for g in range(group(leaf) - 1, -1, -1):
            lo, hi = BOUNDS[g], BOUNDS[g + 1]; pc = fire[:, cur]
            co = (fire[:, lo:hi] & pc[:, None]).sum(0).astype(float)
            lift = (co / ntime) / (pc.mean() * rate[lo:hi] + 1e-12); lift[co < 30] = 0
            cur = lo + int(np.argmax(lift)); chain.append(cur)
        return chain[::-1]
    for leaf in [3153, 3483]:
        print(f"\n=== lineage to G4 specialist {leaf} ===")
        for c in trace(leaf): print(f"  {c:>5}  {fp(c)}")
if __name__ == "__main__":
    main()
