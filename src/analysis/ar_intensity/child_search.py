"""Find all children of parent 99: concepts whose firing is nested in 99 (high containment + lift, rarer)."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; PARENT = 99
def main():
    fire = None
    for r in REGIONS:                                   # concept fires if active (>0.1) in ANY region
        A = np.load(f"{TRACK}/track_matry_{r}.npz")["A_max"].astype(np.float32)
        f = A > THRESH; del A
        fire = f if fire is None else (fire | f)
    ntime = fire.shape[0]; p = fire[:, PARENT]; npar = int(p.sum())
    crate = fire.sum(0).astype(float)
    both = (fire & p[:, None]).sum(0).astype(float)
    contain = both / np.maximum(crate, 1)               # P(99 | c)  -- nesting
    lift = (both / ntime) / ((crate / ntime) * (npar / ntime) + 1e-12)
    ids = np.arange(4096)
    cand = ids[(contain > 0.80) & (crate > 200) & (crate < 0.5 * npar) & (ids != PARENT)]
    order = cand[np.argsort(-lift[cand])]
    print(f"parent 99 fires {npar} steps ({100*npar/ntime:.0f}% of record); {len(cand)} candidate children")
    print(f"{'concept':>7} {'fires':>7} {'%of99fires':>10} {'P(99|c)':>8} {'lift':>6}")
    for c in order[:40]:
        print(f"{c:>7} {int(crate[c]):>7} {100*both[c]/npar:>9.1f}% {contain[c]:>7.2f} {lift[c]:>6.1f}")
    for c in [3153, 3483]:
        print(f" known {c}: P(99|c)={contain[c]:.2f} lift={lift[c]:.1f} fires={int(crate[c])}")
if __name__ == "__main__":
    main()
