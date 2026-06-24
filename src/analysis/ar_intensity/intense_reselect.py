"""Re-select concepts by Pearson with IVT WITHIN the intense regime (and all-AR), to test
whether any Plain-L8 concept tracks high-end intensity rather than just AR onset.
Memory-light: float32, subset rows before any copy (login-node safe)."""
import numpy as np, pandas as pd, gc
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity import concept_ivt_core as C
OUT = "/scratch/euh7ys/climate_xai/concept_ivt"

def main():
    rows = []
    for r in REGIONS:
        d = np.load(f"{OUT}/track_{r}.npz")
        A = d["A"]                                   # float32 (T,4096), ~0.93 GB
        ivt = d["ivt"].astype(np.float64)
        finite = np.isfinite(ivt)
        reg = np.empty(len(ivt), dtype=object); reg[:] = "x"
        reg[finite] = C.ivt_regime(ivt[finite])
        for label, m in [("ar_all", finite & (ivt >= 250.0)),
                         ("intense", finite & (reg == "intense"))]:
            Asub = np.asarray(A[m], dtype=np.float32)        # subset rows only
            rr = C.pearson_cols(Asub, ivt[m].astype(np.float32))
            top = np.argsort(np.nan_to_num(rr, nan=-np.inf))[::-1][:5]
            for rank, ci in enumerate(top):
                rows.append({"region": r, "subset": label, "n": int(m.sum()),
                             "rank": rank + 1, "concept": int(ci), "r": round(float(rr[ci]), 3)})
            del Asub, rr; gc.collect()
        del d, A, ivt, reg; gc.collect()
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/intense_reselect.csv", index=False)
    print(df.to_string()); print("DONE")

if __name__ == "__main__":
    main()
