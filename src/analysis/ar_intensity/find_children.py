"""Children of PARENT in group GRP by co-firing lift (meaningful when the parent is specialized)."""
import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; AR_START = pd.Timestamp("1979-01-01")
REGN = list(REGIONS); MON = "J F M A M J J A S O N D".split(); BOUNDS = [0, 256, 512, 1024, 2048, 4096]
PARENT = int(os.environ.get("PARENT", "411")); GRP = int(os.environ.get("GRP", "2"))
def main():
    lo, hi = BOUNDS[GRP], BOUNDS[GRP + 1]
    fire = None; regcnt = np.zeros((4, 4096)); mocnt = np.zeros((12, 4096)); ivtc = np.zeros(4096); nr = 0
    for ri, r in enumerate(REGN):
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); A = t["A_max"].astype(np.float32); ivt = t["ivt"].astype(float); ti = t["tindex"]
        f = A > THRESH
        fire = f.copy() if fire is None else (fire | f)
        regcnt[ri] = f.sum(0)
        mo = (AR_START + pd.to_timedelta(6 * (ti - 1), unit="h")).month.values
        for mm in range(1, 13): mocnt[mm - 1] += f[mo == mm].sum(0)
        ok = np.isfinite(ivt); Ao = A[ok]; iz = (ivt[ok] - ivt[ok].mean()) / (ivt[ok].std() + 1e-9)
        ivtc += (Ao * iz[:, None]).mean(0) / (Ao.std(0) + 1e-9); nr += 1
        del A, f, t
    ivtc /= nr; ntime = fire.shape[0]; rate = fire.mean(0); pp = fire[:, PARENT]
    def conc(c): tot = c.sum(0); fr = c / np.maximum(tot, 1); return 1 - (-(fr*np.log(fr+1e-12)).sum(0)/np.log(c.shape[0]))
    rcn, mcn = conc(regcnt), conc(mocnt); domreg = np.argmax(regcnt, 0); dommo = np.argmax(mocnt, 0)
    g = np.arange(lo, hi); both = (fire[:, g] & pp[:, None]).sum(0).astype(float)
    lift = (both / ntime) / (pp.mean() * rate[g] + 1e-12); lift[both < 30] = 0
    contain = both / np.maximum(fire[:, g].sum(0), 1)
    order = g[np.argsort(-lift)]
    def fp(c): return f"IVT{ivtc[c]:+.2f} rate={rate[c]*100:4.1f}% spec r/m={rcn[c]:.2f}/{mcn[c]:.2f} {REGN[domreg[c]][:7]:>7}/{MON[dommo[c]]}"
    print(f"PARENT {PARENT} (G{[g for g in range(5) if BOUNDS[g]<=PARENT<BOUNDS[g+1]][0]}):  {fp(PARENT)}\n")
    print(f"--- children in G{GRP} (ranked by co-firing lift with {PARENT}) ---")
    print(f"{'cpt':>5} {'lift':>5} {'P(par|c)':>8}  fingerprint")
    for c in order[:18]:
        print(f"{c:>5} {lift[c-lo]:>5.1f} {contain[c-lo]:>8.2f}  {fp(c)}")
if __name__ == "__main__":
    main()
