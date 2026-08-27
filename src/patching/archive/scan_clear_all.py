"""EXHAUSTIVE scan of every summer timestep. Reads only the window's mesh nodes (mmap subset) for speed.
Ranks by window max IVT. Env LATMIN/LATMAX to set the box."""
import numpy as np, glob, os
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
D = "/scratch/euh7ys/climate_xai/patching"
w = np.load(f"{D}/inject_field_1592.npz")
LA0 = float(os.environ.get("LATMIN", w["lat"].min())); LA1 = float(os.environ.get("LATMAX", w["lat"].max()))
LO0, LO1 = float(w["lon"].min()), float(w["lon"].max())
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
allf = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
def mon(f): return int(f.split("era5_inputs_")[-1][5:7])
sel = [f for f in allf if mon(f) in (6, 7, 8, 9)]
print(f"total summer files: {len(sel)} (of {len(allf)} total); window lat[{LA0:.0f},{LA1:.0f}] lon[{LO0:.0f},{LO1:.0f}]", flush=True)
a0 = np.load(sel[0], mmap_mode="r")
nlat = np.asarray(a0[:, lat_i]).ravel(); nlon = ((np.asarray(a0[:, lon_i]).ravel() + 180) % 360) - 180
box = (nlat >= LA0) & (nlat <= LA1) & (nlon >= LO0) & (nlon <= LO1)
print(f"window nodes {int(box.sum())}", flush=True)
rows = []
for k, f in enumerate(sel):
    a = np.load(f, mmap_mode="r"); ew = np.ascontiguousarray(a[box]).astype(np.float32)
    iv = np.asarray(node_ivt(ew, qi, ui, vi, levels)).ravel()
    s = f.split("era5_inputs_")[-1].replace(".npy", "")
    rows.append((float(iv.max()), float(np.percentile(iv, 99)), s))
    if (k + 1) % 2000 == 0: print(f"  {k+1}/{len(sel)}", flush=True)
nunder = sum(1 for r in rows if r[0] < 250)
print(f"\n{nunder}/{len(rows)} summer timesteps have window-max IVT < 250", flush=True)
print("=== quietest windows (all summer timesteps) ===", flush=True)
for mx, p99, s in sorted(rows)[:40]:
    date, tm = s.split("T"); valid = f"{date}T{tm.replace('-', ':')}"
    tgt = str(np.datetime64(valid) - np.timedelta64(6, "h"))
    print(f"valid {valid}  max {mx:5.0f}  p99 {p99:4.0f}   TARGET={tgt}", flush=True)
print("DONE", flush=True)
