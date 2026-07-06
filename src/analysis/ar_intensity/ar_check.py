"""Definitive AR-mask check: index (date-1979)/6h across the 16 concatenated parts, aggregate node IVT
at AR vs non-AR over many timesteps, extratropical and W_S_America. Base env, SLURM."""
import os, glob, bisect, numpy as np, datetime as DT, xarray as xr
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
from src.analysis.ar_intensity.regions import AR_START, REGIONS
BASE = "/standard/AikyamLab/madelyn/AtmosphericRivers/Intensities"
def main():
    dsets, sizes = [], []
    for i in range(1, 17):
        d = xr.open_dataset(f"{BASE}/ars_part{i}.nc", decode_times=False); dsets.append(d); sizes.append(int(d.time.size))
    cum = np.cumsum([0] + sizes).astype(int); NT = int(cum[-1]); print("NT", NT, flush=True)
    glat = dsets[0].lat.values; glon = dsets[0].lon.values
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat = era0[:, lat_i].astype(float); nlon = era0[:, lon_i].astype(float)
    ilat = np.clip(np.round((nlat - glat[0]) / 0.25).astype(int), 0, len(glat) - 1)
    ilon = np.clip(np.round((nlon - glon[0]) / 0.25).astype(int), 0, len(glon) - 1)
    ext = np.abs(nlat) > 20
    la = REGIONS["W_S_America"]["lat"]; lo = REGIONS["W_S_America"]["lon"][0]
    wsa = (nlat >= la[0]) & (nlat <= la[1]) & (nlon >= lo[0]) & (nlon <= lo[1])
    efiles = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy")); samp = []
    for f in efiles:
        dts = os.path.basename(f).split("era5_inputs_")[-1].replace(".npy", "")
        dt = DT.datetime.strptime(dts, "%Y-%m-%dT%H-%M"); g = int((dt - AR_START).total_seconds() // 21600)
        if 0 <= g < NT: samp.append((g, f))
    rng = np.random.default_rng(0); sub = [samp[i] for i in rng.choice(len(samp), min(300, len(samp)), replace=False)]
    ea, en, wa, wn, cc = [], [], [], [], []
    for g, f in sub:
        part = int(bisect.bisect_right(cum, g) - 1); loc = g - cum[part]
        msl = dsets[part]["event_masks"].isel(time=loc).values
        iv = node_ivt(np.load(f), qi, ui, vi, levels); ar = msl[ilat, ilon] > 0
        if (ext & ar).any(): ea.append(float(iv[ext & ar].mean()))
        if (ext & ~ar).any(): en.append(float(iv[ext & ~ar].mean()))
        if (wsa & ar).any(): wa.append(float(iv[wsa & ar].mean()))
        if (wsa & ~ar).any(): wn.append(float(iv[wsa & ~ar].mean()))
        cc.append(float(np.corrcoef(iv[ext], ar[ext].astype(float))[0, 1]))
    print(f"EXTRATROPICAL (|lat|>20): IVT@AR {np.mean(ea):.0f}  IVT@no-AR {np.mean(en):.0f}  corr {np.nanmean(cc):+.3f}", flush=True)
    print(f"W_S_America:              IVT@AR {np.mean(wa):.0f}  IVT@no-AR {np.mean(wn):.0f}  (n={len(wa)})", flush=True)
if __name__ == "__main__":
    main()
