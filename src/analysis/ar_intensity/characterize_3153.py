"""What is 3153 at its strong firings (where/when), and does it nest in its lineage ancestors?"""
import numpy as np, pandas as pd, collections
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; AR_START = pd.Timestamp("1979-01-01")
NAME = {99:"99  (G0)", 411:"411 (G1)", 941:"941 (G2)", 1838:"1838 (G3)"}
COLS = [99, 411, 941, 1838, 3153]; REGN = list(REGIONS); MON = "J F M A M J J A S O N D".split()
def main():
    Areg = {}; ti = None
    for r in REGN:
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); Areg[r] = t["A_max"][:, COLS].astype(np.float32)
        if ti is None: ti = t["tindex"]
    mo = (AR_START + pd.to_timedelta(6 * (ti - 1), unit="h")).month.values
    pooled = {c: np.stack([Areg[r][:, i] for r in REGN], 1).max(1) for i, c in enumerate(COLS)}
    fire = {c: pooled[c] > THRESH for c in COLS}
    a3153 = pooled[3153]; strong = a3153 >= np.quantile(a3153, 0.99); f3 = fire[3153]
    a3reg = np.stack([Areg[r][:, 4] for r in REGN], 1)
    print(f"3153 fires (>0.1): {int(f3.sum())} steps   strong (top 1%): {int(strong.sum())} steps\n")
    print("WHERE 3153 fires strongest (region with max activation):")
    print("  ", dict(collections.Counter(np.array(REGN)[a3reg[strong].argmax(1)])))
    print("WHEN 3153 fires strongest (month counts):")
    print("  ", {MON[m-1]: int((mo[strong] == m).sum()) for m in range(1, 13)})
    print("\nCONTAINMENT  P(ancestor fires | 3153 fires):")
    for c in [99, 411, 941, 1838]:
        print(f"  {NAME[c]}:  all 3153 firings {100*(fire[c]&f3).sum()/f3.sum():4.0f}%   |   strong only {100*fire[c][strong].mean():4.0f}%")
if __name__ == "__main__":
    main()
