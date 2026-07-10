"""IVT maps for the clamp direction (99, 340, and child 3481), to pair with the saved amp maps."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.patching.run_capstone import metrics
PATCH = "/scratch/euh7ys/climate_xai/patching"; NPZ = f"{PATCH}/plain_L8_sae.npz"
def main():
    S = P.setup(NPZ); meta = np.load(f"{PATCH}/patch_meta.npz", allow_pickle=True); T = str(meta["target_time"])
    inp, tar, frc = P.build_inputs(T, S)
    for cc in (99, 340, 3481):
        d = np.load(f"{PATCH}/delta_clamp_{cc}.npy").astype(np.float32)[:, None, :]
        holder = []
        fwd = P.make_field_capture_forward(jnp.asarray(d), S, holder, inject_step=8, capture_step=15)
        pred = P.run_one(fwd, inp, tar, frc)
        gm, gx, bm, bx, iv = metrics(pred)
        np.save(f"{PATCH}/ivtmap_{cc}_clamp.npy", iv.astype(np.float32))
        print(f"{cc}_clamp: global mean {gm:.1f} | Chile box mean {bm:.1f} max {bx:.1f}", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
