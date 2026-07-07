"""AR content of all 8 nesting concepts vs fixed global IVT thresholds (250=AR, 500=strong AR),
so both families and the 99 children are directly comparable."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
THRESH = 0.1; N = 300; AR = 250.0; STRONG = 500.0
CC = [340, 3481, 3948, 3675, 99, 1454, 3392, 2722]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy")); rng = np.random.default_rng(0)
    sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    ivs = {cc: [] for cc in CC}
    for f in sel:
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for cc in CC:
            fire = acts[:, cc] > THRESH
            if fire.any(): ivs[cc].extend(iv[fire].tolist())
    print(f"AR>= {AR:.0f}, strong-AR>= {STRONG:.0f} kg/m/s\n")
    print(f"{'concept':>7}{'medIVT':>8}{'>=250':>7}{'>=500':>7}{'n':>9}  type")
    for cc in CC:
        v = np.array(ivs[cc])
        if len(v) == 0: print(f"{cc:>7}  (no firings)"); continue
        f250 = 100 * (v >= AR).mean(); f500 = 100 * (v >= STRONG).mean()
        typ = "atmospheric river" if f250 > 50 else "coastal moisture"
        print(f"{cc:>7}{np.median(v):>8.0f}{f250:>6.0f}%{f500:>6.0f}%{len(v):>9}  {typ}")
if __name__ == "__main__":
    main()
