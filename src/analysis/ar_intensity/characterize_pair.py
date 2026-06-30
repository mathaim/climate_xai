"""Characterize a child->parent pair (CHILD fires only when PARENT fires): where/when/how-intense each fires."""
import os, numpy as np, pandas as pd, collections
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; AR_START = pd.Timestamp("1979-01-01")
CHILD = int(os.environ.get("CHILD", "1308")); PARENT = int(os.environ.get("PARENT", "512"))
COLS = [PARENT, CHILD]; REGN = list(REGIONS); MON = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
def main():
    Areg = {}; ti = None; IVT = {}
    for r in REGN:
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); Areg[r] = t["A_max"][:, COLS].astype(np.float32)
        IVT[r] = t["ivt"].astype(float)
        if ti is None: ti = t["tindex"]
    mo = (AR_START + pd.to_timedelta(6 * (ti - 1), unit="h")).month.values
    pooled = {x: np.stack([Areg[r][:, i] for r in REGN], 1).max(1) for i, x in enumerate(COLS)}
    ivtmax = np.stack([IVT[r] for r in REGN], 1).max(1)
    fire = {x: pooled[x] > THRESH for x in COLS}; both = (fire[PARENT] & fire[CHILD]).sum()
    print(f"P(parent {PARENT} | child {CHILD}) = {100*both/fire[CHILD].sum():.0f}%   (child fires {int(fire[CHILD].sum())})")
    print(f"P(child {CHILD} | parent {PARENT}) = {100*both/fire[PARENT].sum():.0f}%   (parent fires {int(fire[PARENT].sum())})\n")
    for x, tag in [(PARENT, "PARENT"), (CHILD, "CHILD")]:
        a = pooled[x]; strong = a >= np.quantile(a, 0.99); i = COLS.index(x)
        areg = np.stack([Areg[r][:, i] for r in REGN], 1)
        dom = collections.Counter(np.array(REGN)[areg[strong].argmax(1)])
        mh = [int((mo[strong] == m).sum()) for m in range(1, 13)]
        ok = np.isfinite(ivtmax[strong])
        print(f"{tag} {x}: strong-firing region {dict(dom)}")
        print(f"    months {dict(zip(MON, mh))}")
        print(f"    region max IVT at strong firing: median {np.median(ivtmax[strong][ok]):.0f}\n")
if __name__ == "__main__":
    main()
