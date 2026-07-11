"""Stage 3: per date, 99 clamp + 99 amp3 + 340 amp3 with L15 capture."""
import glob, os, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
def main():
    S = P.setup(NPZ)
    for f in sorted(glob.glob(f"{PATCH}/x8_*.npy")):
        tag = os.path.basename(f)[3:-4]; T = tag[:13] + ":" + tag[14:]
        inp, tar, frc = P.build_inputs(T, S)
        d99 = np.load(f"{PATCH}/d99_{tag}.npy").astype(np.float32)[:, None, :]
        d340 = np.load(f"{PATCH}/d340_{tag}.npy").astype(np.float32)[:, None, :]
        for name, field in [("99c", d99), ("99a", -2.0*d99), ("340a", -2.0*d340)]:
            h = []
            P.run_one(P.make_field_capture_forward(jnp.asarray(field), S, h, inject_step=8, capture_step=15), inp, tar, frc)
            np.save(f"{PATCH}/l15_{name}_{tag}.npy", np.asarray(h[0], np.float32))
        print(f"{tag}: 3 injections captured", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
