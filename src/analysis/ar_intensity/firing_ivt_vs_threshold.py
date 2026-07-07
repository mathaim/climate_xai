"""Determine weak-AR vs non-AR for 1829/3481/340: distribution of local IVT at their W_S_America
firings vs the pipeline's AR thresholds (p10/p50/p90). SLURM, base env."""
import os, glob, json, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
from src.analysis.ar_intensity.regions import REGIONS
THRESH = 0.1; N = 300; REG = "W_S_America"; CC = [340, 3481, 3948, 3675, 1829]
THR = json.load(open("/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/region_thresholds.json"))[REG]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat = era0[:, lat_i]; nlon = era0[:, lon_i]
    la = REGIONS[REG]["lat"]; lo = REGIONS[REG]["lon"][0]; box = (nlat >= la[0]) & (nlat <= la[1]) & (nlon >= lo[0]) & (nlon <= lo[1])
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy")); rng = np.random.default_rng(0)
    sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    ivs = {cc: [] for cc in CC}
    for f in sel:
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for cc in CC:
            fire = box & (acts[:, cc] > THRESH)
            if fire.any(): ivs[cc].extend(iv[fire].tolist())
    print(f"AR thresholds {REG}: p10={THR['p10']:.0f} p50={THR['p50']:.0f} p90={THR['p90']:.0f}\n")
    print(f"{'concept':>7}{'medIVT':>7}{'%>p10':>7}{'%>p50':>7}{'%>p90':>7}{'nfire':>8}   verdict")
    for cc in CC:
        v = np.array(ivs[cc])
        if len(v) == 0: print(f"{cc:>7}  (no in-region firings)"); continue
        f10 = (v > THR['p10']).mean(); f50 = (v > THR['p50']).mean(); f90 = (v > THR['p90']).mean()
        verdict = "AR-intensity" if f50 > 0.5 else ("weak/fringe-AR" if f10 > 0.4 else "NON-AR (sub-threshold)")
        print(f"{cc:>7}{np.median(v):>7.0f}{100*f10:>6.0f}%{100*f50:>6.0f}%{100*f90:>6.0f}%{len(v):>8}   {verdict}")
if __name__ == "__main__":
    main()
