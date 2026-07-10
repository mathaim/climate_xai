"""Held-out L15 committee/fragment responses vs 20-random floor, with levels."""
import os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching"
T99 = [111, 214, 123, 864, 1269, 2739]; T340 = [3160, 1536, 1675]
def enc(tag, m, c, fmin, frng):
    x = np.load(f"{PATCH}/l15_cap_ho_{tag}.npy").astype(np.float32).reshape(-1, 512)
    if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x)).cpu().numpy()
def main():
    m, c, fmin, frng = load_sae("matry_L15", "cpu"); B = enc("base", m, c, fmin, frng)
    print("baseline levels | " + "  ".join(f"{t}:{B[:,t].sum():.1f}" for t in T99+T340), flush=True)
    rng = np.random.default_rng(0); rand = rng.choice(4096, 20, replace=False)
    for tag, targets in [("99_clamp", T99), ("99_amp3", T99), ("340_amp3", T340)]:
        X = enc(tag, m, c, fmin, frng); d = X - B
        floor = np.median([np.abs(d[:, j]).sum() for j in rand])
        print(f"{tag}: floor {floor:.2f} | " + "  ".join(f"{t}:{d[:,t].sum():+.1f}({np.abs(d[:,t]).sum()/max(floor,1e-9):.0f}x)" for t in targets), flush=True)
if __name__ == "__main__":
    main()
