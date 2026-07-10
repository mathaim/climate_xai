"""Held-out injections with L15 capture: baseline, 99 clamp/amp3, 340 amp3."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
PATCH = "/scratch/euh7ys/climate_xai/patching"; NPZ = f"{PATCH}/plain_L8_sae.npz"
T = "2021-11-15T12:00"
def go(tag, field, S, inp, tar, frc):
    holder = []
    P.run_one(P.make_field_capture_forward(jnp.asarray(field), S, holder, inject_step=8, capture_step=15), inp, tar, frc)
    np.save(f"{PATCH}/l15_cap_ho_{tag}.npy", np.asarray(holder[0], np.float32))
    print(f"{tag}: captured", flush=True)
def main():
    S = P.setup(NPZ); inp, tar, frc = P.build_inputs(T, S)
    d99 = np.load(f"{PATCH}/delta_clamp_99_ho.npy").astype(np.float32)[:, None, :]
    d340 = np.load(f"{PATCH}/delta_clamp_340_ho.npy").astype(np.float32)[:, None, :]
    go("base", np.zeros_like(d99), S, inp, tar, frc)
    go("99_clamp", d99, S, inp, tar, frc)
    go("99_amp3", -2.0*d99, S, inp, tar, frc)
    go("340_amp3", -2.0*d340, S, inp, tar, frc)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
