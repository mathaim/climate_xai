"""Intensity dose-response for the monsoon nesting concepts, binned by local IVT WITHIN the
Indian-Ocean monsoon domain (keyhole-free). Reports mean activation and firing rate per IVT bin."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
SAE = "matry_L8"; N = int(os.environ.get("GLOBAL_N", "400"))
CONCEPTS = [230, 4094, 1986, 3167]
BOX = (-10.0, 30.0, 40.0, 100.0)  # lat0,lat1,lon0,lon1  monsoon domain
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/monsoon_intensity.npz"
BINS = np.array([0, 100, 200, 300, 400, 500, 600, 750, 900, 1100, 1400, 1800, 2500])
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(0); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    nc, nb = len(CONCEPTS), len(BINS) - 1
    actsum = np.zeros((nc, nb)); firesum = np.zeros((nc, nb)); cnt = np.zeros(nb)
    la0, la1, lo0, lo1 = BOX
    for i, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        inb = (era[:, lat_i] >= la0) & (era[:, lat_i] <= la1) & (era[:, lon_i] >= lo0) & (era[:, lon_i] <= lo1)
        iv = node_ivt(era, qi, ui, vi, levels)[inb]
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()[inb][:, CONCEPTS]
        bi = np.clip(np.digitize(iv, BINS) - 1, 0, nb - 1)
        for b in range(nb):
            mk = bi == b
            if mk.any():
                cnt[b] += mk.sum(); actsum[:, b] += acts[mk].sum(0); firesum[:, b] += (acts[mk] > 0.1).sum(0)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    np.savez(OUT, dose=actsum / np.maximum(cnt[None, :], 1), firerate=firesum / np.maximum(cnt[None, :], 1),
             bins=BINS, cnt=cnt, concepts=np.array(CONCEPTS))
    ctr = (BINS[:-1] + BINS[1:]) / 2; print("IVT ctr:", [int(x) for x in ctr], flush=True)
    for k, cc in enumerate(CONCEPTS): print(f"{cc:>5} dose :", " ".join(f"{v:.3f}" for v in actsum[k] / np.maximum(cnt, 1)), flush=True)
    for k, cc in enumerate(CONCEPTS): print(f"{cc:>5} fire%:", " ".join(f"{100*v:.0f}" for v in firesum[k] / np.maximum(cnt, 1)), flush=True)
    print("bin counts:", [int(v) for v in cnt], flush=True)
if __name__ == "__main__":
    main()
