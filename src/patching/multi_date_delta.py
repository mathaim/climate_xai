"""Stage 2: removal fields for 99 and 340 at every captured date (validated formula)."""
import glob, os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"
def main():
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    for f in sorted(glob.glob(f"{PATCH}/x8_*.npy")):
        tag = os.path.basename(f)[3:-4]
        x = np.load(f).astype(np.float32).reshape(-1, 512)
        xs = torch.from_numpy((2.0*(x - fmin)/frng - 1.0).astype(np.float32))
        with torch.no_grad():
            xn = m.normalizer.normalize(xs)
            code = m._apply_topk(xn @ m.W_enc + m.b_enc, m.target_l0)
            s = np.sqrt(512.0)/float(m.normalizer.running_avg)
            for cc in (99, 340):
                zc = code[:, cc].numpy()
                d = (-zc[:, None] * m.W_dec[cc].numpy()[None, :]) * (frng[None, :]/(2.0*s))
                np.save(f"{PATCH}/d{cc}_{tag}.npy", d.astype(np.float32))
                print(f"{tag} c{cc}: fires {(zc>0).sum()}", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
