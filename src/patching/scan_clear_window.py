"""Find a day with IVT below threshold EVERYWHERE in the displayed window (not just the corridor).
Scores the full inject_field window, ranks by window max IVT. No GPU."""
import numpy as np, glob, os
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
D = "/scratch/euh7ys/climate_xai/patching"
w = np.load(f"{D}/inject_field_1592.npz")
LA0 = float(os.environ.get("LATMIN", w["lat"].min())); LA1 = float(os.environ.get("LATMAX", w["lat"].max()))
LO0, LO1 = float(w["lon"].min()), float(w["lon"].max())
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
allf = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
def parse(f):
    s = f.split("era5_inputs_")[-1].replace(".npy", "")
    return int(s[5:7]), int(s[8:10]), s[11:13], s
sel = [f for f in allf if parse(f)[0] in (6, 7, 8, 9)]
if len(sel) > 900: sel = sel[:: max(1, len(sel) // 900)]
print(f"window lat[{LA0:.0f},{LA1:.0f}] lon[{LO0:.0f},{LO1:.0f}], scanning {len(sel)} summer days", flush=True)
rows, box = [], None
for f in sel:
    era = np.load(f)
    if box is None:
        nlat = np.asarray(era[:, lat_i]).ravel(); nlon = ((np.asarray(era[:, lon_i]).ravel() + 180) % 360) - 180
        box = (nlat >= LA0) & (nlat <= LA1) & (nlon >= LO0) & (nlon <= LO1)
        print(f"window nodes {int(box.sum())}", flush=True)
    iv = np.asarray(node_ivt(era, qi, ui, vi, levels)).ravel()[box]
    rows.append((float(iv.max()), float(np.percentile(iv, 99)), float(iv.mean()), parse(f)[3]))
nunder = sum(1 for r in rows if r[0] < 250)
print(f"\n{nunder}/{len(rows)} summer days have window-max IVT < 250", flush=True)
print("=== quietest full windows (ranked by window max IVT) ===", flush=True)
for mx, p99, mn, s in sorted(rows)[:30]:
    date, tm = s.split("T"); valid = f"{date}T{tm.replace('-', ':')}"
    tgt = str(np.datetime64(valid) - np.timedelta64(6, "h"))
    print(f"valid {valid}  max {mx:5.0f}  p99 {p99:4.0f}  mean {mn:4.0f}   TARGET={tgt}", flush=True)
print("DONE", flush=True)
