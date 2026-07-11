"""Stage 4: aggregate committee/fragment responses over dates; mean +- range, train vs held-out."""
import glob, os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"
T99 = [111, 214, 123]; T340 = [1536]
def enc(fp, m, c, fmin, frng):
    x = np.load(fp).astype(np.float32).reshape(-1, 512)
    if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x)).cpu().numpy()
def main():
    m, c, fmin, frng = load_sae("matry_L15", "cpu")
    rng = np.random.default_rng(0); rand = rng.choice(4096, 20, replace=False)
    res = {}
    for f in sorted(glob.glob(f"{PATCH}/l15base_*.npy")):
        tag = os.path.basename(f)[8:-4]; heldout = int(tag[:4]) >= 2018
        B = enc(f, m, c, fmin, frng)
        for name, targets in [("99c", T99), ("99a", T99), ("340a", T340)]:
            X = enc(f"{PATCH}/l15_{name}_{tag}.npy", m, c, fmin, frng); d = X - B
            floor = np.median([np.abs(d[:, j]).sum() for j in rand])
            for t in targets:
                res.setdefault((name, t, heldout), []).append(np.abs(d[:, t]).sum()/max(floor, 1e-9) * np.sign(d[:, t].sum()))
        print(f"{tag} done", flush=True)
    print(f"\n{'cond':>6}{'target':>8}{'set':>10}{'mean ratio':>12}{'range':>22}{'n':>3}")
    for (name, t, ho), v in sorted(res.items()):
        v = np.array(v)
        print(f"{name:>6}{t:>8}{'held-out' if ho else 'train':>10}{v.mean():>12.1f}   [{v.min():.1f}, {v.max():.1f}]{len(v):>3}")
if __name__ == "__main__":
    main()
