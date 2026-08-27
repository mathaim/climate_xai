"""Same-layer dose response: re-encode the edited step-8 activation and read each child's
total code, for 99 -> {1454,3392,2722} and 340 -> {3481,3948,3675}. g=0 clamp, 1 baseline,
2,3 amplify; edit x(g) = x8 + (1-g)*delta_clamp (raw space), matching run_capstone fields."""
import numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file, SAES
import datetime as DT
PATCH = "/scratch/euh7ys/climate_xai/patching"
TS = DT.datetime(2017, 8, 21, 18)
FAM = {99: [1454, 3392, 2722], 340: [3481, 3948, 3675]}
def main():
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    x8 = np.load(act_file(c, TS)).astype(np.float32).reshape(-1, 512)
    for parent, kids in FAM.items():
        d = np.load(f"{PATCH}/delta_clamp_{parent}.npy").astype(np.float32)
        for g in (0, 1, 2, 3):
            x = x8 + (1.0 - g) * d
            xs = 2.0*(x - fmin)/frng - 1.0
            with torch.no_grad(): A = encode(m, c["arch"], torch.from_numpy(xs.astype(np.float32))).numpy()
            row = "  ".join(f"{k}:{A[:,k].sum():.1f}" for k in [parent] + kids)
            print(f"parent {parent} g={g} | " + row, flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
