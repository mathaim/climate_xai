"""Encode capstone L15 captures (matry_L15); dose response of targets vs 20-random floor."""
import os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching"
T99 = [111, 214, 123, 864, 1269, 2739]; T340 = [3160, 1536, 1675]
def enc(tag, m, c, fmin, frng):
    fp = f"{PATCH}/l15_cap_{tag}.npy"
    if not os.path.exists(fp): return None
    x = np.load(fp).astype(np.float32).reshape(-1, 512)
    if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x)).cpu().numpy()
def main():
    m, c, fmin, frng = load_sae("matry_L15", "cpu"); B = enc("cap_base", m, c, fmin, frng)
    rng = np.random.default_rng(0); rand = rng.choice(4096, 20, replace=False)
    print("baseline totals | " + "  ".join(f"{t}:{B[:,t].sum():.1f}" for t in T99 + T340), flush=True)
    for tags, targets in [(["99_clamp","99_amp2","99_amp3"], T99), (["340_amp2","340_amp3"], T340)]:
        for tag in tags:
            X = enc(tag, m, c, fmin, frng)
            if X is None: print(f"missing {tag}"); continue
            d = X - B; floor = np.median([np.abs(d[:, j]).sum() for j in rand])
            row = "  ".join(f"{t}:{d[:,t].sum():+.1f}({np.abs(d[:,t]).sum()/max(floor,1e-9):.0f}x)" for t in targets)
            print(f"{tag}: floor {floor:.2f} | " + row, flush=True)
if __name__ == "__main__":
    main()
