"""Stage 1: capture x8 + l15 baseline for each date in dates_200.txt (chunked, resumable)."""
import os, numpy as np
from src.patching import patch_predict as P
PATCH = "/scratch/euh7ys/climate_xai/patching/multidate"; os.makedirs(PATCH, exist_ok=True)
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
CHUNK = int(os.environ.get("CHUNK", 0)); NCHUNKS = int(os.environ.get("NCHUNKS", 1))
def main():
    DATES = [l.strip() for l in open(f"{PATCH}/dates_200.txt") if l.strip()]
    mine = [T for i, T in enumerate(DATES) if i % NCHUNKS == CHUNK]
    S = P.setup(NPZ)
    for T in mine:
        tag = T.replace(":", "-")
        if os.path.exists(f"{PATCH}/x8_{tag}.npy") and os.path.exists(f"{PATCH}/l15base_{tag}.npy"):
            print(f"{tag}: exists, skip", flush=True); continue
        try:
            inp, tar, frc = P.build_inputs(T, S)
        except Exception as e:
            print(f"{tag}: SKIP (inputs failed: {e})", flush=True); continue
        h8 = []; P.run_one(P.make_captureonly_forward(S, h8, step=8), inp, tar, frc)
        np.save(f"{PATCH}/x8_{tag}.npy", np.asarray(h8[0], np.float32))
        h15 = []; P.run_one(P.make_captureonly_forward(S, h15, step=15), inp, tar, frc)
        np.save(f"{PATCH}/l15base_{tag}.npy", np.asarray(h15[0], np.float32))
        print(f"{tag}: captured", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
