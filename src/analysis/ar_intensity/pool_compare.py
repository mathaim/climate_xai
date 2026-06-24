"""Compare {mean,max} activation pooling x {max,mean} IVT, and re-test intensity saturation
with max-pooled activation. Memory-light: one pooling array at a time, float32."""
import numpy as np, pandas as pd, gc
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity import concept_ivt_core as C
OUT = "/scratch/euh7ys/climate_xai/concept_ivt"
AR_REGIMES = ["weak", "moderate", "intense"]

def tracking_rate(col, ivt, regime):
    za = C.zscore(col); zi = C.zscore(ivt)
    lab = C.classify_corr(C.pointwise_gap(za, zi))
    out = {}
    for reg in AR_REGIMES:
        mr = regime == reg
        hi = int(((lab == "high_corr") & mr).sum()); lo = int(((lab == "low_corr") & mr).sum())
        out[reg] = round(hi / (hi + lo), 3) if (hi + lo) else float("nan")
    return out

def main():
    pair_rows, sat_rows = [], []
    for r in REGIONS:
        d = np.load(f"{OUT}/track_pool_{r}.npz")
        ivt = d["ivt"].astype(np.float64); ivtm = d["ivt_mean"].astype(np.float64)
        ok = np.isfinite(ivt) & np.isfinite(ivtm)
        ivt_o, ivtm_o = ivt[ok], ivtm[ok]; regime = C.ivt_regime(ivt_o)
        sat_cols = {}
        for pool in ["A_mean", "A_max"]:
            A = np.asarray(d[pool][ok], dtype=np.float32)
            for tname, tgt in [("max_ivt", ivt_o), ("mean_ivt", ivtm_o)]:
                rr = C.pearson_cols(A, tgt.astype(np.float32))
                ci = int(np.nanargmax(rr))
                pair_rows.append({"region": r, "pool": pool, "target": tname,
                                  "top_concept": ci, "r": round(float(rr[ci]), 3)})
                if tname == "max_ivt":
                    sat_cols[pool] = (ci, A[:, ci].astype(np.float64).copy())
            del A; gc.collect()
        for pool, (ci, col) in sat_cols.items():
            sat_rows.append({"region": r, "pool": pool, "concept": ci, **tracking_rate(col, ivt_o, regime)})
        del d; gc.collect()
    pd.DataFrame(pair_rows).to_csv(f"{OUT}/pool_pairings.csv", index=False)
    pd.DataFrame(sat_rows).to_csv(f"{OUT}/pool_saturation.csv", index=False)
    print("=== pairing correlations (top-concept r) ==="); print(pd.DataFrame(pair_rows).to_string())
    print("\n=== saturation re-check (tracking rate by regime, best max_ivt concept) ===")
    print(pd.DataFrame(sat_rows).to_string()); print("DONE")

if __name__ == "__main__":
    main()
