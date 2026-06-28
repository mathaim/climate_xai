"""Trace core(parent) -> outer(child) intensity hierarchy pairs in matry_L8.
Global co-firing gives the dependence; per-region intensity |r| gives intensity relevance."""
import numpy as np
COF = "/scratch/euh7ys/climate_xai/cofire/cofire_matry_L8.npz"
CORR = "/home/euh7ys/climate_xai/results/ar_intensity/corr"
REGIONS = ["W_N_America", "W_Europe", "W_S_America", "E_Australia"]
CORE = slice(0, 512)      # G0+G1 parent candidates
OUTER = slice(2048, 4096) # G4 child candidates (most specialized)
FMIN = 50

def grp(i): return "G0" if i < 256 else ("G1" if i < 512 else "G?")

def main():
    d = np.load(COF); C = d["cofire"].astype(np.float64); f = d["fire"].astype(np.float64); N = float(d["nodes"][0])
    R = np.vstack([np.abs(np.load(f"{CORR}/matry_L8_region_magnitude_{r}.npy")) for r in REGIONS])  # (4,4096) |r|
    gen = R.mean(0)                 # intensity-tracking generality
    nreg = (R > 0.2).sum(0)         # #regions a latent is intensity-active in
    base = f / N                    # baseline firing rate
    core = np.arange(4096)[CORE]; rows = []
    for c in np.arange(4096)[OUTER]:
        if f[c] < FMIN: continue
        fa = C[core, c] / f[c]; k = int(np.argmax(fa)); p = core[k]
        if f[p] < FMIN: continue
        fwd = fa[k]; rev = C[p, c] / f[p]; lift = fwd / max(base[p], 1e-9); breadth = f[p] / f[c]
        if fwd < 0.6 or rev > 0.5 * fwd or breadth < 1.3:   # clean hierarchy only (reject splitting)
            continue
        cr = int(np.argmax(R[:, c]))
        rows.append((p, c, grp(p), fwd, rev, lift, breadth, gen[p], int(nreg[p]),
                     R[:, c].max(), REGIONS[cr], int(nreg[c]), gen[p] * lift / (nreg[c] + 1)))
    rows.sort(key=lambda x: -x[-1])
    print(f"{'par':>5}{'chld':>6}{'grp':>4}{'P(p|c)':>8}{'P(c|p)':>8}{'lift':>7}{'brdth':>7}"
          f"{'par|r|':>8}{'p#reg':>6}{'ch|r|':>7}{'ch_region':>14}{'c#reg':>6}")
    for p, c, g, fwd, rev, lift, br, pi, pn, cs, creg, cn, _ in rows[:15]:
        print(f"{p:>5}{c:>6}{g:>4}{fwd:>8.2f}{rev:>8.2f}{lift:>7.1f}{br:>7.1f}"
              f"{pi:>8.3f}{pn:>6d}{cs:>7.2f}{creg:>14}{cn:>6d}")
    print(f"\nDONE: {len(rows)} clean hierarchy pairs")
if __name__ == "__main__":
    main()
