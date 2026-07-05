"""Keyhole-free per-node co-firing: parents (idx 0-1023) x outer children (1024-4095).
Records P(parent|child), P(child|parent), and mean local IVT at firing for every concept,
to pick a tightly-nested AND interpretable pair for patching."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
SAE = "matry_L8"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "200")); NP = 1024
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/global_cofire.npz"
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(2); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    cofire = np.zeros((NP, 4096 - NP), np.float64); pcount = np.zeros(NP); ccount = np.zeros(4096 - NP)
    ivtsum = np.zeros(4096); fcount = np.zeros(4096); used = 0
    for i, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()
        Bf = (acts > THRESH).astype(np.float32); Bp = Bf[:, :NP]; Bc = Bf[:, NP:]
        cofire += Bp.T @ Bc; pcount += Bp.sum(0); ccount += Bc.sum(0)
        ivtsum += (Bf * iv[:, None]).sum(0); fcount += Bf.sum(0); used += 1
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    meanivt = ivtsum / np.maximum(fcount, 1)
    np.savez(OUT, cofire=cofire, pcount=pcount, ccount=ccount, meanivt=meanivt, fcount=fcount, used=used, NP=NP)
    Pp_c = cofire / np.maximum(ccount[None, :], 1)
    bestp = Pp_c.argmax(0); bestP = Pp_c.max(0); ch = np.arange(NP, 4096)
    rows = []
    for j in range(4096 - NP):
        cid, pid = ch[j], bestp[j]
        if ccount[j] < 1000 or pcount[pid] < 2 * ccount[j] or bestP[j] < 0.85: continue
        rows.append((cid, pid, bestP[j], cofire[pid, j] / max(pcount[pid], 1),
                     ccount[j], pcount[pid], meanivt[cid], meanivt[pid]))
    print(f"{'child':>6}{'parent':>7}{'P(p|c)':>7}{'P(c|p)':>7}{'cfire':>8}{'pfire':>9}{'cIVT':>6}{'pIVT':>6}")
    print("--- top 30 by child mean IVT (AR-relevant, tightly nested) ---")
    for r in sorted(rows, key=lambda r: -r[6])[:30]:
        print(f"{r[0]:>6}{r[1]:>7}{r[2]:>7.2f}{r[3]:>7.2f}{int(r[4]):>8}{int(r[5]):>9}{r[6]:>6.0f}{r[7]:>6.0f}")
    print(f"(pairs passing filter: {len(rows)})")
if __name__ == "__main__":
    main()
