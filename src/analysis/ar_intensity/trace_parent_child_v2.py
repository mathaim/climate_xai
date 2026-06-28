"""v2: start from the top intensity-correlated CORE latents (parents), find their outer
children. Uses best-region |r| since matry intensity concepts are region-specific."""
import numpy as np
COF = "/scratch/euh7ys/climate_xai/cofire/cofire_matry_L8.npz"
CORR = "/home/euh7ys/climate_xai/results/ar_intensity/corr"
REGIONS = ["W_N_America", "W_Europe", "W_S_America", "E_Australia"]
FMIN = 50
def grp(i): return "G0" if i < 256 else ("G1" if i < 512 else "G?")
def main():
    d = np.load(COF); C = d["cofire"].astype(np.float64); f = d["fire"].astype(np.float64); N = float(d["nodes"][0])
    R = np.vstack([np.abs(np.load(f"{CORR}/matry_L8_region_magnitude_{r}.npy")) for r in REGIONS])
    rmax = R.max(0); rarg = R.argmax(0); base = f / N
    core = np.arange(512); outer = np.arange(2048, 4096)
    top_parents = core[np.argsort(rmax[core])[::-1][:8]]
    for p in top_parents:
        if f[p] < FMIN: continue
        Ppc = C[p, outer] / np.maximum(f[outer], 1)
        cand = []
        for c in outer[(Ppc > 0.6) & (f[outer] > FMIN)]:
            fwd = C[p, c] / f[c]; rev = C[p, c] / f[p]
            if rev > 0.5 * fwd or f[p] <= f[c]: continue
            cand.append((c, fwd, rev, fwd / max(base[p], 1e-9), f[p] / f[c], rmax[c], REGIONS[rarg[c]]))
        cand.sort(key=lambda x: -x[3])
        print(f"\nPARENT {p} ({grp(p)})  intensity|r|={rmax[p]:.2f} in {REGIONS[rarg[p]]}  "
              f"fires {f[p]/N*100:.2f}%  clean_children={len(cand)}")
        for c, fwd, rev, lift, br, crm, crg in cand[:5]:
            print(f"   child {c}: P(p|c)={fwd:.2f} P(c|p)={rev:.2f} lift={lift:.0f} "
                  f"breadth={br:.0f} child|r|={crm:.2f} in {crg}")
    print("\nDONE")
if __name__ == "__main__":
    main()
