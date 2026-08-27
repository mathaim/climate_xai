"""Stage 4: aggregate concept-99 intervention responses over all captured dates.
Reports, per condition/target/split: mean percent change of total firing, mean
ratio to the 20-random-latent floor, ranges, and n. Skip-tolerant."""
import glob, os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"
T99 = [111, 214, 123]
def enc(fp, m, c, fmin, frng):
    x = np.load(fp).astype(np.float32).reshape(-1, 512)
    if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x)).cpu().numpy()
def main():
    m, c, fmin, frng = load_sae("matry_L15", "cpu")
    rng = np.random.default_rng(0); rand = rng.choice(4096, 20, replace=False)
    pct_res, ratio_res = {}, {}
    allowed = {l.strip().replace(":", "-") for l in open(f"{PATCH}/dates_200.txt") if l.strip()}
    files = [f for f in sorted(glob.glob(f"{PATCH}/l15base_*.npy"))
             if os.path.basename(f)[8:-4] in allowed]
    print(f"aggregating {len(files)} of the 200 sampled dates (pilot dates excluded)", flush=True)
    for f in files:
        tag = os.path.basename(f)[8:-4]; heldout = int(tag[:4]) >= 2018
        B = None
        for name in ("99c", "99a"):
            fp = f"{PATCH}/l15_{name}_{tag}.npy"
            if not os.path.exists(fp): continue
            if B is None: B = enc(f, m, c, fmin, frng)
            X = enc(fp, m, c, fmin, frng); d = X - B
            floor = np.median([np.abs(d[:, j]).sum() for j in rand])
            for t in T99:
                pct_res.setdefault((name, t, heldout), []).append(100.0 * d[:, t].sum() / max(B[:, t].sum(), 1e-9))
                ratio_res.setdefault((name, t, heldout), []).append(np.abs(d[:, t]).sum()/max(floor, 1e-9) * np.sign(d[:, t].sum()))
        print(f"{tag} done", flush=True)
    print(f"\n{'cond':>6}{'target':>8}{'set':>10}{'mean pct':>10}{'pct range':>22}{'mean ratio':>12}{'ratio range':>22}{'n':>4}")
    for k in sorted(pct_res):
        name, t, ho = k
        p = np.array(pct_res[k]); r = np.array(ratio_res[k])
        print(f"{name:>6}{t:>8}{'held-out' if ho else 'train':>10}"
              f"{p.mean():>9.1f}%   [{p.min():.1f}, {p.max():.1f}]"
              f"{r.mean():>12.1f}   [{r.min():.1f}, {r.max():.1f}]{len(p):>4}")
if __name__ == "__main__":
    main()
