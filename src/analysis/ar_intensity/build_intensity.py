"""Stage 3: max IVT over AR cells for every qualifying (timestep, region)."""
import os, numpy as np, pandas as pd
from src.analysis.ar_intensity.ivt_pipeline import (
    load_channel_index, region_node_setup, max_ivt_over_ar, ERA5_DIR, NPZ)
from src.analysis.ar_intensity.regions import index_to_datetime, REGIONS
OUTDIR = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
def main():
    idx, levels, qi, ui, vi, _, _ = load_channel_index()
    setup = region_node_setup(); d = np.load(NPZ)
    masks = {r: d[f"{r}__mask"] for r in REGIONS}
    cov = pd.read_parquet(f"{OUTDIR}/regional_coverage.parquet")
    qual = cov[cov.qualifies]
    rows = []; n = 0; miss = 0
    for ti, grp in qual.groupby("time_index"):
        dt = index_to_datetime(int(ti))
        f = f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy"
        try:
            arr = np.load(f)
        except Exception as e:
            miss += 1
            if miss <= 10: print(f"  [skip {os.path.basename(f)}]: {e}", flush=True)
            continue
        for r in grp.region:
            mi = max_ivt_over_ar(arr, setup[r], masks[r][int(ti)-1], qi, ui, vi, levels)
            rows.append((int(ti), dt, r, mi))
        n += 1
        if n % 2000 == 0: print(f"  {n} timesteps, {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows, columns=["time_index","datetime","region","max_ivt"]).dropna(subset=["max_ivt"])
    df["month"] = df.datetime.dt.month
    df.to_parquet(f"{OUTDIR}/ar_intensity.parquet")
    print(f"saved {len(df)} rows | {miss} timesteps skipped (missing/corrupt era5)")
    for r in REGIONS:
        s = df[df.region==r].max_ivt
        print(f"  {r}: n={len(s)} | maxIVT mean={s.mean():.0f} "
              f"p10={s.quantile(.1):.0f} p50={s.quantile(.5):.0f} p90={s.quantile(.9):.0f}")
if __name__ == "__main__":
    main()
