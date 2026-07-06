"""Pick off/low/high S-Chile timesteps for child 3481; save per-timestep gain base-field (z_c*w)
for the ON cases and a footprint injection field (z_ref*w) for the OFF case. Base env (torch)."""
import os, glob, numpy as np, torch, datetime as DT
import torch.nn.functional as F
from src.analysis.ar_intensity.sae_features import load_sae, encode, SAES
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
DEV = "cpu"; NAME = "matry_L8"; CC = int(os.environ.get("CC", "3481"))
CORE_LL = [float(x) for x in os.environ.get("CORE", "-47,286").split(",")]
RADIUS = float(os.environ.get("RADIUS", "3.0")); MONTHS = [5, 6, 7, 8, 9]
OUT = "/scratch/euh7ys/climate_xai/patching"
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat, nlon = era0[:, lat_i], era0[:, lon_i]
    dlon = np.minimum(np.abs(nlon - CORE_LL[1]), 360 - np.abs(nlon - CORE_LL[1]))
    core = int(np.argmin((nlat - CORE_LL[0])**2 + dlon**2))
    N = np.where(np.sqrt((nlat - nlat[core])**2 + dlon**2) <= RADIUS)[0]
    print("core", core, nlat[core], nlon[core], "| footprint nodes", len(N), flush=True)
    c = SAES[NAME]; files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    cand = [f for f in files if pdt(os.path.basename(f)).month in MONTHS]
    rng = np.random.default_rng(0); scan = [cand[i] for i in rng.choice(len(cand), min(300, len(cand)), replace=False)]
    recs = []
    for f in scan:
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        recs.append((float(node_ivt(era, qi, ui, vi, levels)[core]), f, dt))
    recs.sort(key=lambda r: r[0])
    pick = {"off": recs[int(0.05 * len(recs))], "low": recs[int(0.55 * len(recs))], "high": recs[-1]}
    m, _, fmin, frng = load_sae(NAME, DEV); fmn, frg = torch.tensor(fmin), torch.tensor(frng)
    ra = float(m.normalizer.running_avg.detach().cpu().numpy().item()); s = float(np.sqrt(512) / ra)
    W_enc, b_enc, W_dec = m.W_enc.detach(), m.b_enc.detach(), m.W_dec.detach()
    w = (W_dec[CC] * frg / (2 * s)).numpy().astype(np.float32)
    def code(f):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        xmm = 2 * (torch.from_numpy(x) - fmn) / frg - 1
        z = F.relu((s * xmm) @ W_enc + b_enc); v, ix = torch.topk(z, 32, dim=1); mk = torch.zeros_like(z); mk.scatter_(1, ix, 1.0)
        return (z * mk)[:, CC].numpy()
    zc_hi = code(pick["high"][1]); fires = zc_hi[N] > 0
    z_ref = float(np.median(zc_hi[N][fires])) if fires.any() else float(zc_hi[N].max())
    meta = {"N": N, "w": w, "z_ref": z_ref, "core": core, "core_lat": nlat[core], "core_lon": nlon[core], "cc": CC}
    for tag, (iv, f, dt) in pick.items():
        zc = code(f); meta[f"time_{tag}"] = dt.strftime("%Y-%m-%dT%H:%M"); meta[f"iv_{tag}"] = iv
        base = (zc[:, None] * w[None, :]).astype(np.float32)   # z_c * w  (gain: dx=(g-1)*base)
        np.save(f"{OUT}/base_{tag}.npy", base)
        print(f"{tag}: {dt}  core IVT {iv:.0f}  concept z on footprint mean {zc[N].mean():.3f}  fires@core {zc[core]:.3f}", flush=True)
    inj = np.zeros((len(nlat), 512), np.float32); inj[N] = (z_ref * w)[None, :]   # inject z_ref at footprint
    np.save(f"{OUT}/inj_off.npy", inj)
    np.savez(f"{OUT}/multi_meta.npz", **meta); print("z_ref", z_ref, "| saved multi_meta + fields", flush=True)
if __name__ == "__main__":
    main()
