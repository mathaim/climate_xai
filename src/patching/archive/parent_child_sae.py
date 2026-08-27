"""SAE-internal parent->child test across multiple timesteps: scale parent 340 in the step-8
activation, re-encode, read each child's code at its landfall, aggregated over K timesteps."""
import os, glob, numpy as np, torch, datetime as DT
import torch.nn.functional as F
from src.analysis.ar_intensity.sae_features import load_sae, SAES
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
DEV = "cpu"; NAME = "matry_L8"; PARENT = 340
CHILDREN = {3481: (-47., 286.), 3948: (-38., 289.), 3675: (-35., 288.)}
GAINS = [0.0, 0.5, 1.0, 1.5, 2.0]; MONTHS = [5, 6, 7, 8, 9]; K = int(os.environ.get("K", "6"))
OUT = "/scratch/euh7ys/climate_xai/patching/parent_child_sae.npz"
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat, nlon = era0[:, lat_i], era0[:, lon_i]
    def nearest(lat, lon):
        dlon = np.minimum(np.abs(nlon - lon), 360 - np.abs(nlon - lon)); return int(np.argmin((nlat - lat)**2 + dlon**2))
    cores = {cc: nearest(*ll) for cc, ll in CHILDREN.items()}
    m, c, fmin, frng = load_sae(NAME, DEV); fmn, frg = torch.tensor(fmin), torch.tensor(frng)
    ra = float(m.normalizer.running_avg.detach().cpu().numpy().item()); s = float(np.sqrt(512) / ra)
    W_enc, b_enc, W_dec = m.W_enc.detach(), m.b_enc.detach(), m.W_dec.detach()
    w340 = (W_dec[PARENT] * frg / (2 * s))
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    cand = [f for f in files if pdt(os.path.basename(f)).month in MONTHS]
    rng = np.random.default_rng(1); scan = [cand[i] for i in rng.choice(len(cand), min(250, len(cand)), replace=False)]
    def zcode(x8):
        xmm = 2 * (torch.from_numpy(x8) - fmn) / frg - 1; z = F.relu((s * xmm) @ W_enc + b_enc)
        v, ix = torch.topk(z, 32, dim=1); mk = torch.zeros_like(z); mk.scatter_(1, ix, 1.0); return z * mk
    info = []
    for f in scan:
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        ivn = node_ivt(era, qi, ui, vi, levels); info.append((f, {cc: float(ivn[cores[cc]]) for cc in cores}))
    gi1 = GAINS.index(1.0); save = {"gains": np.array(GAINS)}
    for cc, core in cores.items():
        ranked = sorted(info, key=lambda t: -t[1][cc])[:K]
        cm = np.zeros((K, len(GAINS))); pm = np.zeros((K, len(GAINS))); ivc = []
        for ti, (f, ivd) in enumerate(ranked):
            a = np.load(f, mmap_mode="r"); x8 = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
            z0 = zcode(x8); b340 = (z0[:, PARENT][:, None] * w340[None, :]).numpy().astype(np.float32); ivc.append(ivd[cc])
            for gi, g in enumerate(GAINS):
                zg = zcode(x8 + (g - 1.0) * b340); cm[ti, gi] = float(zg[core, cc]); pm[ti, gi] = float(zg[core, PARENT])
        ok = cm[:, gi1] > 1e-6; crel = cm[ok] / cm[ok, gi1:gi1+1]; prel = pm[ok] / np.maximum(pm[ok, gi1:gi1+1], 1e-6)
        save[f"child_{cc}"] = cm; save[f"parent_{cc}"] = pm
        print(f"\nchild {cc} @ {nlat[core]:.0f},{nlon[core]:.0f}  mean over {int(ok.sum())}/{K} timesteps  (coreIVT {np.mean(ivc):.0f})", flush=True)
        print(f"  {'parent_g':>8}{'z_parent(rel)':>14}{'z_child(rel)':>16}", flush=True)
        for gi, g in enumerate(GAINS):
            print(f"  {g:>8.1f}{prel[:, gi].mean():>14.2f}{crel[:, gi].mean():>12.2f} +/-{crel[:, gi].std():.2f}", flush=True)
    np.savez(OUT, **save); print("\nsaved", OUT, flush=True)
if __name__ == "__main__":
    main()
