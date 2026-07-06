"""Find the AR-mask time alignment: scan integer time offsets and report the mean per-timestep spatial
IVT<->mask correlation. The correct offset spikes to a strong positive correlation. Base env, SLURM."""
import os, glob, numpy as np, datetime as DT
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
from src.analysis.ar_intensity.regions import REGIONS, AR_START
REG = "W_S_America"; MASKS = "/scratch/euh7ys/climate_xai/ar_region_masks.npz"
def inbox(lat, lon):
    la = REGIONS[REG]["lat"]; m = (lat >= la[0]) & (lat <= la[1]); lm = np.zeros_like(m)
    for lo in REGIONS[REG]["lon"]: lm |= (lon >= lo[0]) & (lon <= lo[1])
    return m & lm
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat, nlon = era0[:, lat_i], era0[:, lon_i]
    reg = np.where(inbox(nlat, nlon))[0]
    D = np.load(MASKS); M = D[f"{REG}__mask"]; mlat = D[f"{REG}__lat"]
    lo = REGIONS[REG]["lon"][0]; mlon = np.linspace(lo[0], lo[1], M.shape[2])
    ilat = np.array([int(np.argmin(np.abs(mlat - nlat[n]))) for n in reg])
    ilon = np.array([int(np.argmin(np.abs(mlon - nlon[n]))) for n in reg])
    efiles = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
    rng = np.random.default_rng(1); sel = [efiles[i] for i in rng.choice(len(efiles), 60, replace=False)]
    samp = []
    for f in sel:
        dts = os.path.basename(f).split("era5_inputs_")[-1].replace(".npy", "")
        dt = DT.datetime.strptime(dts, "%Y-%m-%dT%H-%M"); k = int((dt - AR_START).total_seconds() // 21600)
        iv = node_ivt(np.load(f), qi, ui, vi, levels)[reg]; samp.append((k, iv))
    OFFS = sorted(set(list(range(-16, 17)) + [-2920, -1460, -730, -365, -281, -280, -279, -140, -40, 40, 140, 279, 280, 281, 365, 730, 1460, 2920]))
    print(f"{'offset':>8}{'meancorr':>10}{'nts':>6}"); best = (None, -9)
    for off in OFFS:
        cs = []
        for k, iv in samp:
            j = k + off
            if 0 <= j < M.shape[0]:
                mk = (M[j][ilat, ilon] > 0).astype(float)
                if mk.std() > 0: cs.append(float(np.corrcoef(iv, mk)[0, 1]))
        mc = np.mean(cs) if cs else -9
        print(f"{off:>8}{mc:>10.3f}{len(cs):>6}")
        if mc > best[1]: best = (off, mc)
    print(f"\nBEST offset {best[0]}  meancorr {best[1]:.3f}")
if __name__ == "__main__":
    main()
