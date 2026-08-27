"""Offline (torch/base env): encode captured L15 activations with matry_L15 SAE; report each L8
concept's L15-counterpart code baseline vs clamp at the counterpart's baseline peak node."""
import os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching"
CTR = {3481: 3160, 3948: 3392, 3675: 1980}  # L8 -> L15 counterpart (from cross_layer_match)
def enc(tag, m, c, fmin, frng):
    fp = f"{PATCH}/l15_cap_{tag}.npy"
    if not os.path.exists(fp): return None
    x = np.load(fp).astype(np.float32).reshape(-1, 512)
    if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x)).cpu().numpy()
def main():
    m, c, fmin, frng = load_sae("matry_L15", "cpu"); base = enc("baseline", m, c, fmin, frng)
    if base is None: print("no baseline capture"); return
    print(f"{'L8':>5} {'L15ctr':>7} {'peaknode':>9} {'base':>8} {'clamp':>8} {'delta':>8}")
    for l8, l15 in CTR.items():
        clamp = enc(f"clamp_{l8}", m, c, fmin, frng)
        if clamp is None: continue
        pk = int(base[:, l15].argmax()); b = float(base[pk, l15]); cl = float(clamp[pk, l15])
        print(f"{l8:>5} {l15:>7} {pk:>9} {b:>8.3f} {cl:>8.3f} {cl-b:>+8.3f}")
if __name__ == "__main__":
    main()
