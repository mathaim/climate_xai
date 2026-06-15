import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.coverage import cos_lat_coverage
NPZ = "/scratch/euh7ys/climate_xai/ar_region_masks.npz"
OUTDIR = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = np.load(NPZ); rows = []
    for r in REGIONS:
        cov = cos_lat_coverage(d[f"{r}__mask"], d[f"{r}__lat"])
        T = cov.shape[0]; dts = [index_to_datetime(i+1) for i in range(T)]
        rows.append(pd.DataFrame({"time_index": np.arange(1, T+1), "datetime": dts,
            "region": r, "coverage_frac": cov, "qualifies": cov >= 0.5,
            "month": [dt.month for dt in dts]}))
    df = pd.concat(rows, ignore_index=True)
    out = os.path.join(OUTDIR, "regional_coverage.parquet")
    try:
        df.to_parquet(out); saved = out
    except Exception as e:
        out = out.replace(".parquet", ".csv"); df.to_csv(out, index=False); saved = out
        print("(parquet unavailable -> CSV):", e)
    print("\n=== CHECKPOINT: qualifying counts per region (>=50% cos-lat coverage) ===")
    for r in REGIONS:
        s = df[df.region == r]; q = s.coverage_frac; nq = int(s.qualifies.sum())
        print(f"  {r}: qualify={nq}/{len(s)} ({100*nq/len(s):.2f}%) | "
              f"cov p90={q.quantile(.9):.3f} p99={q.quantile(.99):.3f} max={q.max():.3f}")
    print(f"\nsaved {saved}")
if __name__ == "__main__":
    main()
