"""Scan candidate dates for global AR activity: fraction of nodes with IVT >= 250."""
import numpy as np, datetime as DT
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
CANDS = [f"{y}-{m:02d}-15T12-00" for y in (2019, 2020, 2021) for m in (1, 4, 7, 10)]
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    rows = []
    for t in CANDS:
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{t}.npy")
        except FileNotFoundError: print(f"{t}: no ERA5 npy (held-out date, expected)"); continue
        iv = node_ivt(era, qi, ui, vi, levels)
        rows.append((float((iv >= 250).mean()), t))
    for fr, t in sorted(rows): print(f"{t}: AR-node fraction {fr:.4f}")
if __name__ == "__main__":
    main()
