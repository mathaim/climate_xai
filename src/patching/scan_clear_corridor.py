"""Find a day with a genuinely quiet west-coast landfall CORRIDOR (no AR, no filament).
Scores the corridor sub-box only, ranks by corridor max IVT plus 90th pct. No GPU."""
import numpy as np, glob
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
LA0, LA1, LO0, LO1 = 35.0, 52.0, -145.0, -120.0  # offshore approach + coast, N.Cal to BC
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
allf = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
def parse(f):
    s = f.split("era5_inputs_")[-1].replace(".npy", "")
    return int(s[5:7]), int(s[8:10]), s[11:13], s
sel = [f for f in allf if parse(f)[0] in (6, 7, 8, 9)]
if len(sel) > 700: sel = sel[:: max(1, len(sel) // 700)]
print(f"scanning {len(sel)} summer days, corridor lat[{LA0},{LA1}] lon[{LO0},{LO1}]", flush=True)
rows, box = [], None
for f in sel:
    era = np.load(f)
    if box is None:
        nlat = np.asarray(era[:, lat_i]).ravel(); nlon = ((np.asarray(era[:, lon_i]).ravel() + 180) % 360) - 180
        box = (nlat >= LA0) & (nlat <= LA1) & (nlon >= LO0) & (nlon <= LO1)
        print(f"corridor nodes {int(box.sum())}", flush=True)
    w = np.asarray(node_ivt(era, qi, ui, vi, levels)).ravel()[box]
    rows.append((float(w.max()), float(np.percentile(w, 90)), float(w.mean()), parse(f)[3]))
print("\n=== quietest corridors (ranked by corridor max IVT) ===", flush=True)
for mx, p90, mn, s in sorted(rows)[:30]:
    date, tm = s.split("T"); valid = f"{date}T{tm.replace('-', ':')}"
    tgt = str(np.datetime64(valid) - np.timedelta64(6, "h"))
    print(f"valid {valid}  max {mx:5.0f}  p90 {p90:4.0f}  mean {mn:4.0f}   TARGET={tgt}", flush=True)
print("DONE", flush=True)
