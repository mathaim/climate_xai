"""Held-out pass 2: matryoshka removal fields for 99 and 340 from the captured activation.
delta_raw = -code_c (x) W_dec[c] * frng/(2s), s = sqrt(512)/running_avg (the validated path)."""
import numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae
PATCH = "/scratch/euh7ys/climate_xai/patching"
def main():
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    x = np.load(f"{PATCH}/x8_ho.npy").astype(np.float32).reshape(-1, 512)
    xs = torch.from_numpy((2.0*(x - fmin)/frng - 1.0).astype(np.float32))
    with torch.no_grad():
        xn = m.normalizer.normalize(xs)
        code = m._apply_topk(xn @ m.W_enc + m.b_enc, m.target_l0)
        ra = float(m.normalizer.running_avg); s = np.sqrt(512.0)/ra
        for cc in (99, 340):
            zc = code[:, cc].numpy()
            d = (-zc[:, None] * m.W_dec[cc].numpy()[None, :]) * (frng[None, :]/(2.0*s))
            np.save(f"{PATCH}/delta_clamp_{cc}_ho.npy", d.astype(np.float32))
            print(f"concept {cc}: fires {(zc>0).sum()} nodes | delta rms {np.sqrt((d**2).mean()):.4f}", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
