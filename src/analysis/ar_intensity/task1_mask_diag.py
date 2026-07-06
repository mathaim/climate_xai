"""Diagnose the AR-mask handling: node->cell mapping, seasonal cycle (time alignment), and per-timestep
spatial IVT<->mask correlation (a real AR mask must track high IVT). Base env, SLURM."""
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
    print(f"mlat {float(mlat[0])}->{float(mlat[-1])}  mlon(recon) {float(mlon[0])}->{float(mlon[-1])}  mask {M.shape}", flush=True)
    ilat = np.array([int(np.argmin(np.abs(mlat - nlat[n]))) for n in reg])
    ilon = np.array([int(np.argmin(np.abs(mlon - nlon[n]))) for n in reg])
    print("=== node->cell mapping (first 5 region nodes) ===")
    for j in range(5):
        n = reg[j]; print(f"  node lat {nlat[n]:.2f} lon {nlon[n]:.2f}  ->  cell lat {mlat[ilat[j]]:.2f} lon {mlon[ilon[j]]:.2f}")
    cov = np.zeros(13); cnt = np.zeros(13)
    for k in range(0, M.shape[0], 17):
        mo = (AR_START + DT.timedelta(hours=6 * k)).month; cov[mo] += float((M[k] > 0).mean()); cnt[mo] += 1
    print("=== AR coverage by month (expect austral-winter May-Sep peak) ===")
    print("  " + " ".join(f"{m}:{cov[m]/max(cnt[m],1):.2f}" for m in range(1, 13)))
    efiles = sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
    rng = np.random.default_rng(0); sel = [efiles[i] for i in rng.choice(len(efiles), min(150, len(efiles)), replace=False)]
    corrs = []; ivtAR = []; ivtNO = []
    for f in sel:
        dts = os.path.basename(f).split("era5_inputs_")[-1].replace(".npy", "")
        dt = DT.datetime.strptime(dts, "%Y-%m-%dT%H-%M"); k = int((dt - AR_START).total_seconds() // 21600)
        if k < 0 or k >= M.shape[0]: continue
        iv = node_ivt(np.load(f), qi, ui, vi, levels)[reg]; mk = (M[k][ilat, ilon] > 0).astype(float)
        if mk.std() > 0: corrs.append(float(np.corrcoef(iv, mk)[0, 1]))
        if mk.sum() > 0: ivtAR.append(float(iv[mk > 0].mean()))
        if (mk == 0).sum() > 0: ivtNO.append(float(iv[mk == 0].mean()))
    print(f"=== spatial IVT<->mask corr (per-timestep mean over {len(corrs)}): {np.mean(corrs):+.3f} ===")
    print(f"   mean IVT at mask-nodes {np.mean(ivtAR):.0f} vs non-mask {np.mean(ivtNO):.0f}")
if __name__ == "__main__":
    main()
