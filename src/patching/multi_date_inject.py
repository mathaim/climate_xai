"""Stage 3: per date, 99 clamp (beta=0) + 99 amp (beta=3) with L15 capture. Chunked, resumable."""
import glob, os, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
CHUNK = int(os.environ.get("CHUNK", 0)); NCHUNKS = int(os.environ.get("NCHUNKS", 1))
def main():
    files = sorted(glob.glob(f"{PATCH}/d99_*.npy"))
    mine = [f for i, f in enumerate(files) if i % NCHUNKS == CHUNK]
    S = P.setup(NPZ)
    for f in mine:
        tag = os.path.basename(f)[4:-4]; T = tag[:13] + ":" + tag[14:]
        if os.path.exists(f"{PATCH}/l15_99c_{tag}.npy") and os.path.exists(f"{PATCH}/l15_99a_{tag}.npy"):
            print(f"{tag}: exists, skip", flush=True); continue
        inp, tar, frc = P.build_inputs(T, S)
        d99 = np.load(f).astype(np.float32)[:, None, :]
        for name, field in [("99c", d99), ("99a", -2.0*d99)]:
            h = []
            P.run_one(P.make_field_capture_forward(jnp.asarray(field), S, h, inject_step=8, capture_step=15), inp, tar, frc)
            np.save(f"{PATCH}/l15_{name}_{tag}.npy", np.asarray(h[0], np.float32))
        print(f"{tag}: 2 injections captured", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
