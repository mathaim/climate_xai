"""Regional clear-day scan on the mesh (no jax/graphcast): box = NE-Pacific window of the 1592 maps.
Reuses find_quiet_day machinery (node_ivt on era5_inputs npy). Reports box max IVT + AR-node frac."""
import numpy as np, glob
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
D = "/scratch/euh7ys/climate_xai/patching"
win = np.load(f"{D}/inject_field_1592.npz"); la, lo = win["lat"], win["lon"]
LA0, LA1, LO0, LO1 = float(la.min()), float(la.max()), float(lo.min()), float(lo.max())
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
allf = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
def parse(f):
    s = f.split("era5_inputs_")[-1].replace(".npy", "")  # YYYY-MM-DDTHH-MM
    return int(s[5:7]), int(s[8:10]), s[11:13], s
sel = [f for f in allf if (lambda p: p[0] in (7, 8, 9) and p[2] == "12" and p[1] % 2 == 1)(parse(f))]
if len(sel) > 320: sel = sel[:: max(1, len(sel) // 320)]
print(f"available summer-noon files: {len([f for f in allf if parse(f)[0] in (7,8,9)])}, scanning {len(sel)}", flush=True)
rows, box = [], None
for f in sel:
    era = np.load(f)
    if box is None:
        nlat = np.asarray(era[:, lat_i]).ravel(); nlon = ((np.asarray(era[:, lon_i]).ravel() + 180) % 360) - 180
        box = (nlat >= LA0) & (nlat <= LA1) & (nlon >= LO0) & (nlon <= LO1)
        print(f"box nodes {int(box.sum())}  lat[{LA0:.1f},{LA1:.1f}] lon[{LO0:.1f},{LO1:.1f}]", flush=True)
    iv = np.asarray(node_ivt(era, qi, ui, vi, levels)).ravel()
    w = iv[box]
    rows.append((float(w.max()), float((w >= 250).mean()), parse(f)[3]))
print(f"\n=== cleanest days in NE-Pacific box ({len(rows)} scanned) ===", flush=True)
for mx, fr, s in sorted(rows)[:25]:
    date, tm = s.split("T"); valid = f"{date}T{tm.replace('-', ':')}"
    tgt = str(np.datetime64(valid) - np.timedelta64(6, "h"))
    print(f"valid {valid}  boxMaxIVT {mx:6.0f}  ARfrac {fr:.3f}   TARGET={tgt}", flush=True)
print("DONE", flush=True)
