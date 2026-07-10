"""Held-out pass 3: removal and amplification maps for 99 and 340."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.patching.run_capstone import metrics
PATCH = "/scratch/euh7ys/climate_xai/patching"; NPZ = f"{PATCH}/plain_L8_sae.npz"
T = "2021-11-15T12:00"
def main():
    S = P.setup(NPZ); inp, tar, frc = P.build_inputs(T, S)
    for cc in (99, 340):
        d = np.load(f"{PATCH}/delta_clamp_{cc}_ho.npy").astype(np.float32)[:, None, :]
        for tag, field in [(f"ho_{cc}_clamp", d), (f"ho_{cc}_amp3", -2.0*d)]:
            pred = P.run_one(P.make_field_forward(jnp.asarray(field), S, 8), inp, tar, frc)
            gm, gx, bm, bx, iv = metrics(pred)
            np.save(f"{PATCH}/ivtmap_{tag}.npy", iv.astype(np.float32))
            print(f"{tag}: global mean {gm:.1f} max {gx:.1f}", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
