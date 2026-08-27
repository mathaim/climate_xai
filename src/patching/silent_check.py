import numpy as np, glob
from src.analysis.ar_intensity.ivt_pipeline import ERA5_DIR
D = "/scratch/euh7ys/climate_xai"
d = np.load(f"{D}/concept_ivt/track_pool_W_N_America.npz")
z = d['A_max'][:, 1592]; iv = d['ivt']; ti = d['tindex']
files = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
print("n files", len(files), " n track", len(z), " tindex max", int(ti.max()), flush=True)
dates = np.array([files[int(t)].split("era5_inputs_")[-1].replace(".npy", "") for t in ti])
mask = np.char.startswith(dates, "1979-06-09")
print("\n1979-06-09 (date, regional ivt, z_1592):", flush=True)
for i in np.where(mask)[0]: print("  ", dates[i], round(float(iv[i]), 0), round(float(z[i]), 3), flush=True)
c = np.load(f"{D}/patching/clear_maps_7906.npz"); lat = c['lat']; m = (lat >= 25) & (lat <= 58)
print("window baseline IVT 25-58N:", round(float(np.nanmax(c['baseline'][m])), 0),
      " truth:", round(float(np.nanmax(c['truth'][m])), 0), flush=True)
s = (z < 1e-6); idx = np.where(s)[0]; oo = idx[np.argsort(iv[idx])]
print("\nsilent-1592 days:", int(s.sum()), "- clearest by regional IVT (date, ivt):", flush=True)
for i in oo[:15]: print("  ", dates[i], round(float(iv[i]), 0), flush=True)
