"""Uniform cross-layer tracking for ALL L8 concepts (parents + children, geo + AR).
Counterpart = argmax firing-Jaccard (same rule for every concept). Reports the full Balcells
et al. suite: Pearson (continuous acts), Jaccard, Sufficiency P(L15|L8), Necessity P(L8|L15)."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
N = 200; D = "/scratch/euh7ys/climate_xai/concept_ivt"
GROUPS = [("cross_layer.npz", [340, 3481, 3948, 3675]), ("cross_layer_99.npz", [99, 1454, 3392, 2722])]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy", ""), "%Y-%m-%dT%H-%M")
def enc(f, m, c, fmin, frng):
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
    if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
def main():
    pairs = []  # [l8, l15, J, suff, nec]
    for fn, ccs in GROUPS:
        d = np.load(f"{D}/{fn}"); wsa = list(d["wsa"])
        for cc in ccs:
            k = wsa.index(cc); j = int(np.argmax(d["jac"][k])); both = d["both"][k, j]
            pairs.append([cc, j, float(d["jac"][k, j]), float(both / d["cnt8"][k]), float(both / d["cnt15"][j])])
    l8s = [p[0] for p in pairs]; l15s = [p[1] for p in pairs]
    m8, c8, fmin8, frng8 = load_sae("matry_L8", "cpu"); m15, c15, fmin15, frng15 = load_sae("matry_L15", "cpu")
    dtmap = lambda dd: {pdt(os.path.basename(f)): f for f in glob.glob(f"{dd}/layer*_*.npy")}
    f8, f15 = dtmap(c8["act"]), dtmap(c15["act"]); shared = sorted(set(f8) & set(f15))
    rng = np.random.default_rng(0); sel = [shared[i] for i in rng.choice(len(shared), min(N, len(shared)), replace=False)]
    P = len(pairs); n = 0; sx = np.zeros(P); sy = np.zeros(P); sxy = np.zeros(P); sxx = np.zeros(P); syy = np.zeros(P)
    for i, dt in enumerate(sel):
        a8 = enc(f8[dt], m8, c8, fmin8, frng8); a15 = enc(f15[dt], m15, c15, fmin15, frng15)
        X = a8[:, l8s]; Y = a15[:, l15s]
        n += X.shape[0]; sx += X.sum(0); sy += Y.sum(0); sxy += (X * Y).sum(0); sxx += (X * X).sum(0); syy += (Y * Y).sum(0)
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(sel)}", flush=True)
    cov = sxy - sx * sy / n; vx = sxx - sx * sx / n; vy = syy - sy * sy / n
    pear = cov / np.sqrt(vx * vy + 1e-12)
    print(f"\n{'L8':>5} {'L15':>5} {'Pearson':>8} {'Jaccard':>8} {'Suff':>6} {'Nec':>6}")
    for i, p in enumerate(pairs):
        print(f"{p[0]:>5} {p[1]:>5} {pear[i]:>8.3f} {p[2]:>8.3f} {p[3]:>6.3f} {p[4]:>6.3f}")
if __name__ == "__main__":
    main()
