"""Held-out L8 children dose: re-encode x8_ho + (1-g)*delta_ho, read children codes."""
import numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching"
FAM = {99: [1454, 3392, 2722], 340: [3481, 3948, 3675]}
def main():
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    x8 = np.load(f"{PATCH}/x8_ho.npy").astype(np.float32).reshape(-1, 512)
    for parent, kids in FAM.items():
        d = np.load(f"{PATCH}/delta_clamp_{parent}_ho.npy").astype(np.float32)
        for g in (0, 1, 2, 3):
            xs = 2.0*((x8 + (1.0-g)*d) - fmin)/frng - 1.0
            with torch.no_grad(): A = encode(m, c["arch"], torch.from_numpy(xs.astype(np.float32))).numpy()
            print(f"parent {parent} g={g} | " + "  ".join(f"{k}:{A[:,k].sum():.1f}" for k in [parent]+kids), flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
