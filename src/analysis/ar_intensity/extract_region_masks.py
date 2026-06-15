"""One-time: extract the 4 region subsets of the AR mask (class_masks==2) from
the 16 ars_part*.nc files into one compact .npz. Reads class_masks in sequential
time-chunks (fast over Ceph) instead of lat/lon-subset seeks."""
import argparse, glob, os, re
import numpy as np, xarray as xr
from src.analysis.ar_intensity.regions import REGIONS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_dir", default="/standard/AikyamLab/madelyn/AtmosphericRivers/Intensities")
    ap.add_argument("--out", default="/scratch/euh7ys/climate_xai/ar_region_masks.npz")
    ap.add_argument("--chunk", type=int, default=300)
    a = ap.parse_args()
    parts = sorted(glob.glob(os.path.join(a.ar_dir, "ars_part*.nc")),
                   key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    ds0 = xr.open_dataset(parts[0]); lat = ds0.lat.values; lon = ds0.lon.values; ds0.close()
    ridx = {}
    for r, cfg in REGIONS.items():
        la = np.where((lat >= cfg["lat"][0]) & (lat <= cfg["lat"][1]))[0]
        lo = np.concatenate([np.where((lon >= x) & (lon <= y))[0] for x, y in cfg["lon"]])
        ridx[r] = (la, lo)
    store = {r: [] for r in REGIONS}
    for p in parts:
        ds = xr.open_dataset(p); nt = ds.sizes["time"]
        for i in range(0, nt, a.chunk):
            blk = ds.class_masks.isel(time=slice(i, i + a.chunk)).values   # sequential read
            ar = (blk == 2).astype(np.uint8)
            for r, (la, lo) in ridx.items():
                store[r].append(ar[:, la][:, :, lo])
        ds.close(); print(f"done {os.path.basename(p)}", flush=True)
    save = {}
    for r in REGIONS:
        save[f"{r}__mask"] = np.concatenate(store[r], axis=0)
        save[f"{r}__lat"] = lat[ridx[r][0]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **save)
    print("saved", a.out, {r: save[f"{r}__mask"].shape for r in REGIONS}, flush=True)

if __name__ == "__main__":
    main()
