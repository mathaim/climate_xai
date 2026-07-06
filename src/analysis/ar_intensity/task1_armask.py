"""Ground-truth AR check (W_S_America): for each concept, firing rate under the AR detection mask vs
without it. Uses the aligned region masks (ordinal 6-hourly from 1979-01-01). Base env, SLURM."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
from src.analysis.ar_intensity.regions import REGIONS, AR_START
SAE = "matry_L8"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "400")); REG = "W_S_America"
MASKS = "/scratch/euh7ys/climate_xai/ar_region_masks.npz"
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/task1_armask.npz"
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
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
    print("mask", M.shape, M.dtype, "unique", np.unique(M[::9000]), "mlat", float(mlat[0]), float(mlat[-1]), flush=True)
    ilat = np.array([int(np.argmin(np.abs(mlat - nlat[n]))) for n in reg])
    ilon = np.array([int(np.argmin(np.abs(mlon - nlon[n]))) for n in reg])
    m, c, fmin, frng = load_sae(SAE, "cpu")
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(0); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    fire_AR = np.zeros(4096); fire_no = np.zeros(4096); act_AR = np.zeros(4096); cntAR = 0; cntNO = 0; used = 0
    for f in sel:
        dt = pdt(os.path.basename(f)); k = int((dt - AR_START).total_seconds() // 21600)
        if k < 0 or k >= M.shape[0]: continue
        ar = M[k][ilat, ilon] > 0
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): A = np.maximum(encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()[reg], 0)
        B = A > THRESH
        fire_AR += B[ar].sum(0); fire_no += B[~ar].sum(0); act_AR += A[ar].sum(0)
        cntAR += int(ar.sum()); cntNO += int((~ar).sum()); used += 1
        if used % 50 == 0: print(f"  {used}/{len(sel)}", flush=True)
    np.savez(OUT, fire_AR=fire_AR, fire_no=fire_no, act_AR=act_AR, cntAR=cntAR, cntNO=cntNO, used=used)
    pAR = fire_AR / max(cntAR, 1); pNO = fire_no / max(cntNO, 1); enr = pAR / np.maximum(pNO, 1e-9)
    print(f"\nused {used} | AR-node samples {cntAR}, no-AR {cntNO}", flush=True)
    print(f"{'con':>6}{'P(f|AR)':>10}{'P(f|noAR)':>11}{'enrich':>9}")
    for cc in [1829, 3481, 340]:
        print(f"{cc:>6}{pAR[cc]:>10.4f}{pNO[cc]:>11.4f}{enr[cc]:>9.1f}")
    ok = (fire_AR + fire_no) > 100; order = [c for c in np.argsort(-enr) if ok[c]][:15]
    print("\nTop concepts by AR enrichment:")
    for c in order:
        print(f"{c:>6}{pAR[c]:>10.4f}{pNO[c]:>11.4f}{enr[c]:>9.1f}")
if __name__ == "__main__":
    main()
