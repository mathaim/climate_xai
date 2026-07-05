"""Global intensity dose-response: mean concept activation binned by LOCAL node IVT (all mesh nodes, no region boxes)."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
SAE = "matry_L8"
CONCEPTS = [int(x) for x in os.environ.get("CONCEPTS", "99,3153,3483").split(",")]
N = int(os.environ.get("GLOBAL_N", "300"))
OUT = os.environ.get("OUT", "/scratch/euh7ys/climate_xai/concept_ivt/global_intensity.npz")
BINS = np.array([0, 100, 200, 300, 400, 500, 600, 750, 900, 1100, 1400, 1800, 2500])
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(0); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    nc, nb = len(CONCEPTS), len(BINS) - 1
    actsum = np.zeros((nc, nb)); cnt = np.zeros(nb)
    for i, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels); bi = np.clip(np.digitize(iv, BINS) - 1, 0, nb - 1)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()[:, CONCEPTS]
        for b in range(nb):
            mk = bi == b
            if mk.any():
                cnt[b] += mk.sum(); actsum[:, b] += acts[mk].sum(0)
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    np.savez(OUT, dose=actsum / np.maximum(cnt[None, :], 1), bins=BINS, cnt=cnt, concepts=np.array(CONCEPTS))
    print("saved", OUT, flush=True)
if __name__ == "__main__":
    main()
