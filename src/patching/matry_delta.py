"""Precompute raw-activation delta fields to CLAMP matry-L8 concepts at processor step 8.
Base env (torch). Picks the strongest-Arabian-Sea JJA timestep, saves delta[nnode,512] per concept + meta."""
import os, glob, numpy as np, torch, datetime as DT
import torch.nn.functional as F
from src.analysis.ar_intensity.sae_features import load_sae, encode, SAES
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
DEV = "cpu"; NAME = "matry_L8"; CLAMP = [230, 4094]
CORE = (12.0, 67.0)  # Arabian Sea core lat, lon (0-360 -> 67)
OUTDIR = "/scratch/euh7ys/climate_xai/patching"; os.makedirs(OUTDIR, exist_ok=True)
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat, nlon = era0[:, lat_i], era0[:, lon_i]
    core = np.argmin((nlat - CORE[0])**2 + (np.minimum(np.abs(nlon - CORE[1]), np.abs(nlon - 360 - CORE[1])))**2)
    print("core node", core, "at", nlat[core], nlon[core], flush=True)
    c = SAES[NAME]; files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    jja = [f for f in files if pdt(os.path.basename(f)).month in (6, 7, 8)]
    rng = np.random.default_rng(0); scan = [jja[i] for i in rng.choice(len(jja), min(150, len(jja)), replace=False)]
    best, bestiv = None, -1
    for f in scan:
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)[core]
        if iv > bestiv: bestiv, best = iv, f
    dt = pdt(os.path.basename(best)); print(f"BEST monsoon ts: {dt}  core IVT {bestiv:.0f}", flush=True)
    m, _, fmin, frng = load_sae(NAME, DEV)
    fmn = torch.tensor(fmin); frg = torch.tensor(frng)
    s = float(np.sqrt(512) / m.normalizer.running_avg.detach().cpu().numpy().item())
    W_enc, b_enc, W_dec = m.W_enc.detach(), m.b_enc.detach(), m.W_dec.detach()
    a = np.load(best, mmap_mode="r"); x8 = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
    x8t = torch.from_numpy(x8); x_mm = 2 * (x8t - fmn) / frg - 1
    z = F.relu((s * x_mm) @ W_enc + b_enc)
    vals, ix = torch.topk(z, 32, dim=1); mask = torch.zeros_like(z); mask.scatter_(1, ix, 1.0); z = z * mask
    rep = encode(m, "matry", x_mm)  # get_acts on min-max input, as in analysis
    for cc in CLAMP:
        zc = z[:, cc]
        chk = (zc * W_dec[cc].norm() / s)
        err = float((chk - rep[:, cc]).abs().max())
        delta = -(zc[:, None] * W_dec[cc][None, :]) * frg[None, :] / (2 * s)
        np.save(f"{OUTDIR}/delta_clamp_{cc}.npy", delta.numpy().astype(np.float32))
        print(f"concept {cc}: fires {int((zc>0).sum())} nodes | consistency max|err|={err:.2e} | delta rms={float(delta.pow(2).mean().sqrt()):.4f}", flush=True)
    np.savez(f"{OUTDIR}/monsoon_meta.npz", target_time=dt.strftime("%Y-%m-%dT%H:%M"),
             core=core, core_lat=nlat[core], core_lon=nlon[core], x8=x8, s=s, clamp=np.array(CLAMP))
    print("saved deltas + meta to", OUTDIR, flush=True)
if __name__ == "__main__":
    main()
