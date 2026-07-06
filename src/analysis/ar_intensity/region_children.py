"""Find outer children concentrated in one of the 4 AR regions (activation-mass fraction, keyhole-free)
and contained in a broad parent. Cross-references global_cofire.npz for containment."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
from src.analysis.ar_intensity.regions import REGIONS
SAE = "matry_L8"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "300"))
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/region_children.npz"
COF = "/scratch/euh7ys/climate_xai/concept_ivt/global_cofire.npz"
RN = list(REGIONS.keys())
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def inbox(lat, lon, reg):
    la = REGIONS[reg]["lat"]; m = (lat >= la[0]) & (lat <= la[1]); lm = np.zeros_like(m)
    for lo in REGIONS[reg]["lon"]: lm |= (lon >= lo[0]) & (lon <= lo[1])
    return m & lm
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat, nlon = era0[:, lat_i], era0[:, lon_i]
    masks = np.stack([inbox(nlat, nlon, r) for r in RN], 1)
    print("region node counts:", masks.sum(0), flush=True)
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(4); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    asum = np.zeros(4096); rsum = np.zeros((4096, 4)); fc = np.zeros(4096); iv_s = np.zeros(4096)
    for i, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = np.maximum(encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy(), 0)
        asum += acts.sum(0)
        for r in range(4): rsum[:, r] += acts[masks[:, r]].sum(0)
        B = acts > THRESH; fc += B.sum(0); iv_s += (B * iv[:, None]).sum(0)
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    rfrac = rsum / np.maximum(asum[:, None], 1e-9); meaniv = iv_s / np.maximum(fc, 1)
    np.savez(OUT, rfrac=rfrac, asum=asum, fc=fc, meaniv=meaniv, regions=np.array(RN))
    C = np.load(COF); cofire, pcount, ccount, NP = C["cofire"], C["pcount"], C["ccount"], int(C["NP"])
    Pp_c = cofire / np.maximum(ccount[None, :], 1); bestp = Pp_c.argmax(0); bestP = Pp_c.max(0)
    for r, rn in enumerate(RN):
        print(f"\n=== {rn}: outer children concentrated here, contained in a broad parent ===", flush=True)
        print(f"{'child':>6}{'rfrac':>6}{'IVT':>5}{'fires':>7}{'parent':>7}{'P(p|c)':>7}{'par_rf':>7}")
        cand = []
        for j in range(4096 - NP):
            cid = NP + j
            if rfrac[cid, r] < 0.3 or fc[cid] < 300: continue
            pid = bestp[j]
            if bestP[j] < 0.80 or pcount[pid] < 2 * ccount[j]: continue
            cand.append((cid, rfrac[cid, r], meaniv[cid], fc[cid], pid, bestP[j], rfrac[pid, r]))
        for t in sorted(cand, key=lambda t: -t[1])[:12]:
            print(f"{t[0]:>6}{t[1]:>6.2f}{t[2]:>5.0f}{int(t[3]):>7}{t[4]:>7}{t[5]:>7.2f}{t[6]:>7.2f}")
        if not cand: print("  (none pass filter)")
if __name__ == "__main__":
    main()
