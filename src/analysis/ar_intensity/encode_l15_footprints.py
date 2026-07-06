"""Strong-firing footprints of the L15 counterparts of the WSA concepts (for the L8-vs-L15 figure)."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
SAE = "matry_L15"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "300"))
CONCEPTS = [3160, 3392, 1980, 1536, 1675, 756]   # L15 child-matches + parent's fragmented pieces
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/characterize_l15.npz"
conv = lambda x: x - 360 if x > 180 else x
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(float); nlon = np.array([conv(x) for x in era0[:, lon_i]], float)
    m, c, fmin, frng = load_sae(SAE, "cpu")
    files = sorted(glob.glob(f"{c['act']}/layer0015_*.npy"))
    rng = np.random.default_rng(0); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    ev = {cc: {"node": [], "act": []} for cc in CONCEPTS}
    for i, f in enumerate(sel):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for cc in CONCEPTS:
            aa = acts[:, cc]; mk = np.where(aa > THRESH)[0]
            ev[cc]["node"].append(mk); ev[cc]["act"].append(aa[mk])
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    save = {"nlat": nlat, "nlon": nlon, "concepts": np.array(CONCEPTS)}
    for cc in CONCEPTS:
        save[f"n_{cc}"] = np.concatenate(ev[cc]["node"]); save[f"a_{cc}"] = np.concatenate(ev[cc]["act"])
    np.savez(OUT, **save); print("saved", OUT, flush=True)
if __name__ == "__main__":
    main()
