"""Stage 1: per date, capture step-8 activation and step-15 baseline activation."""
import numpy as np
from src.patching import patch_predict as P
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"
import os; os.makedirs(PATCH, exist_ok=True)
DATES = ["1985-01-15T12:00","1995-07-01T00:00","2005-04-10T06:00","2015-10-20T18:00",
         "2019-01-15T12:00","2020-04-10T06:00","2021-07-15T00:00","2021-11-15T12:00"]
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
def main():
    S = P.setup(NPZ)
    for T in DATES:
        tag = T.replace(":", "-")
        inp, tar, frc = P.build_inputs(T, S)
        h8 = []; P.run_one(P.make_captureonly_forward(S, h8, step=8), inp, tar, frc)
        np.save(f"{PATCH}/x8_{tag}.npy", np.asarray(h8[0], np.float32))
        h15 = []; P.run_one(P.make_captureonly_forward(S, h15, step=15), inp, tar, frc)
        np.save(f"{PATCH}/l15base_{tag}.npy", np.asarray(h15[0], np.float32))
        print(f"{tag}: captured x8 + l15 baseline", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
