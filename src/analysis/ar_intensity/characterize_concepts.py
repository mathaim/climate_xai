"""Keyhole-free identity of candidate nested-pair concepts: strong-firing location/spread, IVT, season."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
SAE = "matry_L8"; THRESH = 0.1; N = int(os.environ.get("GLOBAL_N", "400"))
CONCEPTS = [int(x) for x in os.environ.get("CONCEPTS", "1829,3481,340").split(",")]
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/characterize_ar.npz"
conv = lambda x: x - 360 if x > 180 else x
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(float); nlon = np.array([conv(x) for x in era0[:, lon_i]], float); nnode = era0.shape[0]
    m, c, fmin, frng = load_sae(SAE, dev)
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy"))
    rng = np.random.default_rng(3); sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    ev = {cc: {"node": [], "act": [], "ivt": [], "mon": []} for cc in CONCEPTS}
    for i, f in enumerate(sel):
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).cpu().numpy()
        for cc in CONCEPTS:
            aa = acts[:, cc]; mk = np.where(aa > THRESH)[0]
            ev[cc]["node"].append(mk); ev[cc]["act"].append(aa[mk]); ev[cc]["ivt"].append(iv[mk]); ev[cc]["mon"].append(np.full(len(mk), dt.month))
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    ntop = max(1, nnode // 100); MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    save = {"nlat": nlat, "nlon": nlon, "concepts": np.array(CONCEPTS)}
    print(f"{'con':>5}{'nfire':>7}{'IVT':>5}{'IVTs':>6}{'cont':>6}  strong(medlat,lon) lat[10-90] lon[10-90]  months")
    for cc in CONCEPTS:
        node = np.concatenate(ev[cc]["node"]); act = np.concatenate(ev[cc]["act"])
        ivt = np.concatenate(ev[cc]["ivt"]); mon = np.concatenate(ev[cc]["mon"])
        fc = np.bincount(node, minlength=nnode); cont = np.sort(fc)[::-1][:ntop].sum() / max(fc.sum(), 1)
        s = act >= np.quantile(act, 0.99); la, lo = nlat[node[s]], nlon[node[s]]
        tm = [MON[k] for k in np.argsort(-np.bincount(mon[s], minlength=13))[:3]]
        print(f"{cc:>5}{len(act):>7}{np.median(ivt):>5.0f}{np.median(ivt[s]):>6.0f}{cont:>6.2f}  ({np.median(la):>4.0f},{np.median(lo):>4.0f}) [{np.percentile(la,10):>4.0f},{np.percentile(la,90):>3.0f}] [{np.percentile(lo,10):>5.0f},{np.percentile(lo,90):>4.0f}]  {tm}")
        save[f"n_{cc}"] = node; save[f"a_{cc}"] = act; save[f"i_{cc}"] = ivt; save[f"m_{cc}"] = mon
    np.savez(OUT, **save); print("saved", OUT)
if __name__ == "__main__":
    main()
