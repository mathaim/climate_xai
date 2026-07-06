"""Task 1 foundation (keyhole-free, matry_L8): over many timesteps, for the 3 S-Chile children,
compute co-firing vs ALL concepts, per-concept firing rate, W_S_America activation-mass fraction, and
region-restricted IVT tuning. Stores per-timestep arrays for bootstrap CIs. Feeds AR parent selection."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
from src.analysis.ar_intensity.regions import REGIONS
SAE = "matry_L8"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "400")); REG = "W_S_America"
CHILDREN = [int(x) for x in os.environ.get("CHILDREN", "664,1829,3481").split(",")]
BINS = np.array([0, 100, 200, 300, 400, 500, 600, 750, 900, 1100, 1400, 1800, 2500])
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/task1_foundation.npz"
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def inbox(lat, lon):
    la = REGIONS[REG]["lat"]; m = (lat >= la[0]) & (lat <= la[1]); lm = np.zeros_like(m)
    for lo in REGIONS[REG]["lon"]: lm |= (lon >= lo[0]) & (lon <= lo[1])
    return m & lm
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat, nlon = era0[:, lat_i], era0[:, lon_i]
    mask = inbox(nlat, nlon); nnode = era0.shape[0]; print("region nodes", int(mask.sum()), flush=True)
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(0); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    nb = len(BINS) - 1; nc = len(CHILDREN); Nt = len(sel)
    cofire_ts = np.zeros((Nt, nc, 4096), np.float32); childfire_ts = np.zeros((Nt, nc), np.float32); rate_ts = np.zeros((Nt, 4096), np.float32)
    actsum = np.zeros(4096); regmass = np.zeros(4096)
    tun_act = np.zeros((4096, nb)); tun_fire = np.zeros((4096, nb)); tun_cnt = np.zeros(nb); used = 0
    for ti, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): A = np.maximum(encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy(), 0)
        B = (A > THRESH).astype(np.float32); rate_ts[ti] = B.sum(0)
        for ci, ch in enumerate(CHILDREN):
            cm = B[:, ch]; childfire_ts[ti, ci] = cm.sum(); cofire_ts[ti, ci] = (B * cm[:, None]).sum(0)
        actsum += A.sum(0); regmass += A[mask].sum(0)
        ivr = iv[mask]; Ar = A[mask]; Br = B[mask]; bi = np.clip(np.digitize(ivr, BINS) - 1, 0, nb - 1)
        for b in range(nb):
            mk = bi == b
            if mk.any(): tun_cnt[b] += mk.sum(); tun_act[:, b] += Ar[mk].sum(0); tun_fire[:, b] += Br[mk].sum(0)
        used += 1
        if (ti + 1) % 50 == 0: print(f"  {ti+1}/{Nt}", flush=True)
    np.savez(OUT, cofire_ts=cofire_ts, childfire_ts=childfire_ts, rate_ts=rate_ts, actsum=actsum, regmass=regmass,
             tun_act=tun_act, tun_fire=tun_fire, tun_cnt=tun_cnt, bins=BINS, children=np.array(CHILDREN),
             nnode=nnode, region_nodes=int(mask.sum()), used=used)
    print("saved", OUT, "used", used, flush=True)
if __name__ == "__main__":
    main()
