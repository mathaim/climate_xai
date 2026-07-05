"""Screen strictly-contained concepts (from region_scan) for AR relevance: mean local IVT at firing,
containment, parent-99 co-firing, peak location. Find a region-contained AR concept, keyhole-free."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
SAE = "matry_L8"; THRESH = 0.1; PARENT = 99; N = int(os.environ.get("GLOBAL_N", "500"))
RS = "/scratch/euh7ys/climate_xai/concept_ivt/region_scan.npz"
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/cand_screen.npz"
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    R = np.load(RS); conc, tot, plat, plon = R["conc"], R["tot"], R["peaklat"], R["peaklon"]
    cand = np.where((conc >= 0.7) & (tot >= 300) & (tot <= 8000))[0]
    cand = cand[np.argsort(-conc[cand])][:150]; ncand = len(cand); print("candidates:", ncand, flush=True)
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nnode = era0.shape[0]
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(1); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    fc = np.zeros((ncand, nnode)); sivt = np.zeros(ncand); nfire = np.zeros(ncand); cof = np.zeros(ncand)
    for i, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()
        p99 = acts[:, PARENT] > THRESH; fr = acts[:, cand] > THRESH
        fc += fr.T.astype(float); nfire += fr.sum(0); sivt += (fr * iv[:, None]).sum(0); cof += (fr & p99[:, None]).sum(0)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    meanivt = sivt / np.maximum(nfire, 1)
    ntop = max(1, nnode // 100)
    contain = np.partition(fc, -ntop, axis=1)[:, -ntop:].sum(1) / np.maximum(nfire, 1)
    cofrate = cof / np.maximum(nfire, 1)
    np.savez(OUT, cand=cand, nfire=nfire, meanivt=meanivt, contain=contain, cofrate=cofrate,
             peaklat=plat[cand], peaklon=plon[cand])
    order = np.argsort(-meanivt)
    print(f"{'concept':>7}{'nfire':>7}{'meanIVT':>8}{'contain':>8}{'cof99':>6}{'plat':>7}{'plon':>7}")
    for j in order[:35]:
        print(f"{cand[j]:>7}{int(nfire[j]):>7}{meanivt[j]:>8.0f}{contain[j]:>8.2f}{cofrate[j]:>6.2f}{plat[cand[j]]:>7.1f}{plon[cand[j]]:>7.1f}")
if __name__ == "__main__":
    main()
