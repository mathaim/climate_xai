"""Layer-8 children responses across the 200 sampled dates. CPU only:
edited step-8 activation = x8 + field, exactly (the edit is additive at step 8)."""
import os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"
KIDS = [1454, 3392, 2722]
def main():
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    fmn = torch.from_numpy(np.asarray(fmin, np.float32)); frg = torch.from_numpy(np.asarray(frng, np.float32))
    def enc(x):
        xs = (2.0*(torch.from_numpy(x) - fmn)/frg - 1.0)
        with torch.no_grad(): return encode(m, c["arch"], xs).cpu().numpy()
    allowed = {l.strip().replace(":", "-") for l in open(f"{PATCH}/dates_200.txt") if l.strip()}
    res = {}
    done = 0
    for tag in sorted(allowed):
        fx, fd = f"{PATCH}/x8_{tag}.npy", f"{PATCH}/d99_{tag}.npy"
        if not (os.path.exists(fx) and os.path.exists(fd)): continue
        heldout = int(tag[:4]) >= 2018
        x = np.load(fx).astype(np.float32).reshape(-1, 512)
        d = np.load(fd).astype(np.float32)
        zb = enc(x); zc = enc(x + d); za = enc(x - 2.0*d)
        for k in KIDS:
            b = max(zb[:, k].sum(), 1e-9)
            res.setdefault(("99c", k, heldout), []).append(100.0*(zc[:, k].sum()-b)/b)
            res.setdefault(("99a", k, heldout), []).append(100.0*(za[:, k].sum()-b)/b)
        done += 1
        if done % 20 == 0: print(f"{done} dates", flush=True)
    print(f"\n{'cond':>6}{'child':>8}{'set':>10}{'mean pct':>10}{'range':>24}{'n':>4}")
    for key in sorted(res):
        name, k, ho = key; p = np.array(res[key])
        print(f"{name:>6}{k:>8}{'held-out' if ho else 'train':>10}{p.mean():>9.1f}%   [{p.min():.1f}, {p.max():.1f}]{len(p):>4}")
if __name__ == "__main__":
    main()
