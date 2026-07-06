"""Verify the canonical AR product (ars_part1.nc): index by position (6-hourly from 1979-01-01), map mesh
nodes to the real global grid, and confirm the mask tracks region IVT before using it. Base env, SLURM."""
import os, glob, numpy as np, datetime as DT, xarray as xr
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
from src.analysis.ar_intensity.regions import REGIONS, AR_START
REG = "W_S_America"; NC = "/standard/AikyamLab/madelyn/AtmosphericRivers/Intensities/ars_part1.nc"
def inbox(lat, lon):
    la = REGIONS[REG]["lat"]; m = (lat >= la[0]) & (lat <= la[1]); lm = np.zeros_like(m)
    for lo in REGIONS[REG]["lon"]: lm |= (lon >= lo[0]) & (lon <= lo[1])
    return m & lm
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat, nlon = era0[:, lat_i], era0[:, lon_i]
    reg = np.where(inbox(nlat, nlon))[0]
    d = xr.open_dataset(NC, decode_times=False); glat = d.lat.values; glon = d.lon.values; nt = d.time.size
    ig = np.array([int(np.argmin(np.abs(glat - nlat[n]))) for n in reg])
    jg = np.array([int(np.argmin(np.abs(glon - nlon[n]))) for n in reg])
    print(f"part1 nt={nt}  covers pos 0..{nt-1}  (~{AR_START + DT.timedelta(hours=6*(nt-1))})", flush=True)
    em = d["event_masks"]; cm = d["class_masks"]
    efiles = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
    cand = []
    for f in efiles:
        dts = os.path.basename(f).split("era5_inputs_")[-1].replace(".npy", "")
        dt = DT.datetime.strptime(dts, "%Y-%m-%dT%H-%M"); p = int((dt - AR_START).total_seconds() // 21600)
        if 0 <= p < nt: cand.append((f, p))
    rng = np.random.default_rng(3); sub = [cand[i] for i in rng.choice(len(cand), min(90, len(cand)), replace=False)]
    print(f"candidate activation timesteps in part1 window: {len(cand)}, testing {len(sub)}", flush=True)
    for name, var in [("event_masks", em), ("class_masks", cm)]:
        cs = []; ar = []; no = []
        for f, p in sub:
            iv = node_ivt(np.load(f), qi, ui, vi, levels)[reg]
            mk = (var.isel(time=p).values[ig, jg] > 0).astype(float)
            if mk.std() > 0: cs.append(float(np.corrcoef(iv, mk)[0, 1]))
            if mk.sum() > 0: ar.append(float(iv[mk > 0].mean()))
            if (mk == 0).sum() > 0: no.append(float(iv[mk == 0].mean()))
        print(f"{name}: n={len(cs)} meancorr={np.mean(cs):+.3f}  IVT ar={np.mean(ar):.0f} no={np.mean(no):.0f}", flush=True)
