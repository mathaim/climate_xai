"""Cross-layer concept trace: match matry-L8 W_S_America concepts to matry-L15 latents by
per-node/per-time firing overlap (Jaccard). Node-aligned, keyhole-free, shared timesteps."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode, SAES
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
DEV = "cpu"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "200"))
WSA = [int(x) for x in os.environ.get("CONCEPTS","340,3481,3948,3675").split(",")]
OUT = os.environ.get("OUT","/scratch/euh7ys/climate_xai/concept_ivt/cross_layer.npz")
conv = lambda x: x - 360 if x > 180 else x
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def firing(f, m, c, fmin, frng):
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
    if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
    with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(DEV)).cpu().numpy()
    return acts > THRESH
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i]; nlon = np.array([conv(x) for x in era0[:, lon_i]]); nnode = era0.shape[0]
    m8, c8, fmin8, frng8 = load_sae("matry_L8", DEV)
    m15, c15, fmin15, frng15 = load_sae("matry_L15", DEV)
    dtmap = lambda d: {pdt(os.path.basename(f)): f for f in glob.glob(f"{d}/layer*_*.npy")}
    f8, f15 = dtmap(c8["act"]), dtmap(c15["act"])
    shared = sorted(set(f8) & set(f15)); print("shared timesteps:", len(shared), flush=True)
    rng = np.random.default_rng(0); sel = [shared[i] for i in rng.choice(len(shared), min(N, len(shared)), replace=False)]
    nw = len(WSA); both = np.zeros((nw, 4096)); cnt8 = np.zeros(nw); cnt15 = np.zeros(4096)
    fc15n = np.zeros((nnode, 4096), np.float32); fc8n = np.zeros((nnode, nw), np.float32)
    for i, dt in enumerate(sel):
        B8 = firing(f8[dt], m8, c8, fmin8, frng8)[:, WSA]
        B15 = firing(f15[dt], m15, c15, fmin15, frng15)
        both += B8.T.astype(float) @ B15.astype(float)
        cnt8 += B8.sum(0); cnt15 += B15.sum(0); fc15n += B15; fc8n += B8
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    jac = both / np.maximum(cnt8[:, None] + cnt15[None, :] - both, 1)
    np.savez(OUT, jac=jac, both=both, cnt8=cnt8, cnt15=cnt15, wsa=np.array(WSA))
    for k, cc in enumerate(WSA):
        pk8 = fc8n[:, k].argmax()
        print(f"\nL8 {cc}  fires {int(cnt8[k])} node-events  peak({nlat[pk8]:.0f},{nlon[pk8]:.0f})  -> best L15 matches:", flush=True)
        for j in np.argsort(-jac[k])[:4]:
            pk = fc15n[:, j].argmax()
            print(f"   L15 {j:>4}  Jaccard {jac[k,j]:.3f}  overlap {int(both[k,j]):>6}  L15fires {int(cnt15[j]):>7}  peak({nlat[pk]:.0f},{nlon[pk]:.0f})", flush=True)
if __name__ == "__main__":
    main()
