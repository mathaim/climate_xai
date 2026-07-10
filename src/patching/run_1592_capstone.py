"""1592 with L15 capture: clamp during the BC AR (necessity) and write into clear air
(sufficiency), reading the plain-L15 successor 2251 downstream."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
PATCH = "/scratch/euh7ys/climate_xai/patching"
NPZ = f"{PATCH}/plain_L8_sae.npz"
AR_T, CLEAR_T = "2021-11-15T12:00", "2021-07-15T00:00"
def go(tag, T, alpha, cls, S):
    inp, tar, frc = P.build_inputs(T, S); holder = []
    fwd = P.make_capture_forward(jnp.asarray(alpha), S, holder, cls, edit_step=8, capture_step=15)
    P.run_one(fwd, inp, tar, frc)
    if holder: np.save(f"{PATCH}/l15p_cap_{tag}.npy", np.asarray(holder[0], np.float32))
    print(f"{tag}: captured {np.asarray(holder[0]).shape if holder else None}", flush=True)
def main():
    S = P.setup(NPZ)
    z = np.zeros(4096, np.float32)
    go("ar_base",   "2021-11-15T12:00", z, P.AlphaCaptureInjector, S)
    a = z.copy(); a[1592] = -1.0
    go("ar_clamp1592", "2021-11-15T12:00", a, P.AlphaCaptureInjector, S)
    go("clear_base", "2021-07-15T00:00", z, P.AddCaptureInjector, S)
    b = z.copy(); b[1592] = 1.0
    go("clear_add1592", "2021-07-15T00:00", b, P.AddCaptureInjector, S)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
