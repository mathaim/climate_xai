"""Full-record region IVT: max & mean over ALL region nodes for every timestep (dry + AR).
Reuses the existing local-ERA5 IVT pipeline; no Zarr restream."""
import numpy as np, pandas as pd
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, region_node_setup, ERA5_DIR
from src.analysis.ar_intensity.ivt import ivt
from src.analysis.ar_intensity.regions import index_to_datetime, REGIONS
OUTDIR = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
TOTAL = 56700

def main():
    _, levels, qi, ui, vi, _, _ = load_channel_index()
    setup = region_node_setup()
    rows = []; miss = 0
    for ti in range(1, TOTAL + 1):
        dt = index_to_datetime(ti)
        f = f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy"
        try:
            arr = np.load(f, mmap_mode="r")
        except Exception:
            miss += 1; continue
        for r in REGIONS:
            rn = np.ascontiguousarray(arr[setup[r]["nodes"]])
            iv = ivt(rn[:, qi], rn[:, ui], rn[:, vi], levels)
            rows.append((ti, dt, r, float(iv.max()), float(iv.mean())))
        if ti % 2000 == 0:
            print(ti, "/", TOTAL, "rows", len(rows), "miss", miss, flush=True)
    df = pd.DataFrame(rows, columns=["time_index", "datetime", "region", "max_ivt", "mean_ivt"])
    df["month"] = df.datetime.dt.month
    df.to_parquet(f"{OUTDIR}/ar_intensity_full.parquet")
    print(f"saved {len(df)} rows | {miss} timesteps missing")
    for r in REGIONS:
        s = df[df.region == r]
        print(f"  {r}: n={len(s)} maxIVT p50={s.max_ivt.median():.0f} meanIVT p50={s.mean_ivt.median():.0f}")

if __name__ == "__main__":
    main()
