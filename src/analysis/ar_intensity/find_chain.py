"""Extend the 512(G2)->1308(G3) containment pair to a G1 parent and a G4 child, by containment."""
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; THRESH = 0.1; REGN = list(REGIONS); BOUNDS = [0,256,512,1024,2048,4096]
def grp(i):
    for g in range(5):
        if BOUNDS[g] <= i < BOUNDS[g+1]: return g
def main():
    fire = None; regcnt = np.zeros((4, 4096))
    for ri, r in enumerate(REGN):
        A = np.load(f"{TRACK}/track_matry_{r}.npz")["A_max"].astype(np.float32); f = A > THRESH; del A
        fire = f if fire is None else (fire | f); regcnt[ri] = f.sum(0)
    ntime = fire.shape[0]; rate = fire.sum(0).astype(float); domreg = np.array(REGN)[regcnt.argmax(0)]; ff = fire.astype(np.float32)
    def fp(c): return f"G{grp(c)}  {domreg[c][:9]:>9}  rate={rate[c]/ntime*100:4.0f}%  fires={int(rate[c])}"
    f512 = ff[:, 512]; g1 = np.arange(256, 512)
    Pg1 = (ff[:, g1] * f512[:, None]).sum(0) / rate[512]                      # P(g1 | 512)
    m1 = (rate[g1] > rate[512]) & (rate[g1] < 0.5 * ntime); gp = 256 + int(np.argmax(np.where(m1, Pg1, -1)))
    f1308 = ff[:, 1308]; g4 = np.arange(2048, 4096)
    Pg4 = (ff[:, g4] * f1308[:, None]).sum(0) / np.maximum(rate[g4], 1)        # P(1308 | g4)
    m4 = (rate[g4] < rate[1308]) & (rate[g4] > 200); gc = 2048 + int(np.argmax(np.where(m4, Pg4, -1)))
    print("CONTAINMENT CHAIN  (each fires only when the one above fires):")
    print(f"  G1 {gp:>4}: {fp(gp)}      [512 fires only when {gp} fires:  P={Pg1[gp-256]:.2f}]")
    print(f"  G2  512: {fp(512)}")
    print(f"  G3 1308: {fp(1308)}")
    print(f"  G4 {gc:>4}: {fp(gc)}      [{gc} fires only when 1308 fires:  P={Pg4[gc-2048]:.2f}]")
    print(f"\nCHAIN_IDS={gp},512,1308,{gc}")
if __name__ == "__main__":
    main()
