"""Examine the AR mask and directly compare node IVT vs AR-presence at each node+timestep (global)."""
import os, glob, numpy as np, datetime as DT, xarray as xr
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
from src.analysis.ar_intensity.regions import AR_START
NC = "/standard/AikyamLab/madelyn/AtmosphericRivers/Intensities/ars_part1.nc"
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    efiles = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
    era0 = np.load(efiles[0]); nlat = era0[:, lat_i].astype(float); nlon = era0[:, lon_i].astype(float)
    print("=== NODE COORDS ===")
    print(f"  lat range [{nlat.min():.2f},{nlat.max():.2f}]  first 5: {np.round(nlat[:5],2)}")
    print(f"  lon range [{nlon.min():.2f},{nlon.max():.2f}]  first 5: {np.round(nlon[:5],2)}")
    d = xr.open_dataset(NC, decode_times=False); glat = d.lat.values; glon = d.lon.values; nt = d.time.size
    print(f"=== MASK GRID === glat [{glat[0]},{glat[-1]}] glon [{glon[0]},{glon[-1]}] nt {nt}")
    ilat = np.clip(np.round((nlat - glat[0]) / 0.25).astype(int), 0, len(glat) - 1)
    ilon = np.clip(np.round((nlon - glon[0]) / 0.25).astype(int), 0, len(glon) - 1)
    print(f"  map check: node(lat {nlat[0]:.2f},lon {nlon[0]:.2f}) -> cell(lat {glat[ilat[0]]:.2f},lon {glon[ilon[0]]:.2f})")
    em = d["event_masks"]; got = 0
    for f in efiles:
        dts = os.path.basename(f).split("era5_inputs_")[-1].replace(".npy", "")
        dt = DT.datetime.strptime(dts, "%Y-%m-%dT%H-%M"); p = int((dt - AR_START).total_seconds() // 21600)
        if not (0 <= p < nt): continue
        iv = node_ivt(np.load(f), qi, ui, vi, levels); msl = em.isel(time=p).values; ar = msl[ilat, ilon] > 0
        hi = iv > 500
        print(f"\n{dt}  pos {p} | IVT[min {iv.min():.0f} max {iv.max():.0f} mean {iv.mean():.0f}]")
        print(f"  ALL-NODE: AR frac {ar.mean():.3f} | IVT@AR {iv[ar].mean():.0f} vs @no-AR {iv[~ar].mean():.0f} | corr {np.corrcoef(iv, ar.astype(float))[0,1]:+.3f}")
        print(f"  of high-IVT nodes (>500): {int((hi & ar).sum())} are AR, {int((hi & ~ar).sum())} are NOT AR")
        got += 1
        if got >= 5: break
if __name__ == "__main__":
    main()
