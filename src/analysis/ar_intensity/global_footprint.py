"""Global per-node FIRING FREQUENCY footprint + monthly firing climatology, for given matryoshka concepts."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
SAE = "matry_L8"; THRESH = 0.1
CONCEPTS = [int(x) for x in os.environ.get("CONCEPTS", "99,3153,3483").split(",")]
N = int(os.environ.get("GLOBAL_N", "480"))
OUT = os.environ.get("FOOT_OUT", "/scratch/euh7ys/climate_xai/concept_ivt/global_footprint.npz")
def parse_dt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(np.float32); nlon = era0[:, lon_i].astype(np.float32); nnode = era0.shape[0]
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(0); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    print(f"sampling {len(sel)} of {len(files)} (all months)", flush=True)
    nc = len(CONCEPTS); freq = np.zeros((nc, nnode)); monthsum = np.zeros((nc, 12)); monthcnt = np.zeros(12); cnt = 0
    for i, f in enumerate(sel):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(nnode, -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad():
            acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()
        mo = parse_dt(os.path.basename(f)).month
        for k, cc in enumerate(CONCEPTS):
            fk = acts[:, cc] > THRESH; freq[k] += fk; monthsum[k, mo - 1] += fk.mean()
        monthcnt[mo - 1] += 1; cnt += 1
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    np.savez(OUT, foot=(freq / max(cnt, 1)).astype(np.float32),
             monthclim=(monthsum / np.maximum(monthcnt, 1)).astype(np.float32),
             nlat=nlat, nlon=nlon, concepts=np.array(CONCEPTS))
    print("saved", OUT, "cnt", cnt, flush=True)
if __name__ == "__main__":
    main()
