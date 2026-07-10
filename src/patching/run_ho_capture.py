"""Held-out (Nov 2021) pass 1: capture step-8 activation and the baseline forecast IVT map."""
import numpy as np
from src.patching import patch_predict as P
from src.patching.run_capstone import metrics
PATCH = "/scratch/euh7ys/climate_xai/patching"; NPZ = f"{PATCH}/plain_L8_sae.npz"
T = "2021-11-15T12:00"
def main():
    S = P.setup(NPZ); inp, tar, frc = P.build_inputs(T, S); holder = []
    pred = P.run_one(P.make_captureonly_forward(S, holder, step=8), inp, tar, frc)
    gm, gx, bm, bx, iv = metrics(pred)
    np.save(f"{PATCH}/x8_ho.npy", np.asarray(holder[0], np.float32))
    np.save(f"{PATCH}/ivtmap_ho_base.npy", iv.astype(np.float32))
    print(f"captured x8 {np.asarray(holder[0]).shape}; baseline global mean {gm:.1f} max {gx:.1f}", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
