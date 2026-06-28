"""What do parent 99's children fire for, if not region/intensity? Characterize each
child's top firing events by region, season (month), IVT level, parent co-activation, year."""
import numpy as np
from collections import Counter
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"
REG = list(REGIONS); PARENT = 99; CHILDREN = [3153, 3483, 3392]
def main():
    big = {c: {k: [] for k in ["act","reg","ivt","mon","yr","pa"]} for c in CHILDREN}
    for r in REG:
        t = np.load(f"{TRACK}/track_matry_{r}.npz")
        ivt = t["ivt"].astype(float); ok = np.isfinite(ivt)
        ivt = ivt[ok]; mon = t["month"][ok]; ti = t["tindex"][ok]
        yr = 1979 + (ti - 1) // 1461
        pa = t["A_max"][ok, PARENT].astype(float)
        for c in CHILDREN:
            ca = t["A_max"][ok, c].astype(float)
            big[c]["act"].append(ca); big[c]["reg"].append(np.full(len(ca), r[:5]))
            big[c]["ivt"].append(ivt); big[c]["mon"].append(mon); big[c]["yr"].append(yr); big[c]["pa"].append(pa)
        del t
    for c in CHILDREN:
        g = {k: np.concatenate(v) for k, v in big[c].items()}
        top = np.argsort(g["act"])[::-1][:300]
        print(f"\nCHILD {c}: top-300 firing events")
        print("  region :", dict(Counter(g['reg'][top])))
        print("  month  :", dict(sorted(Counter(g['mon'][top]).items())))
        print(f"  IVT    : median {np.median(g['ivt'][top]):.0f} (AR thr 250), "
              f"frac>=250 {(g['ivt'][top]>=250).mean():.0%}")
        print(f"  parent99 act at child-firing: {np.median(g['pa'][top]):.2f}  vs overall median {np.median(g['pa']):.2f}")
        print(f"  years  : {g['yr'][top].min()}-{g['yr'][top].max()}, distinct {len(set(g['yr'][top].tolist()))}/39")
    print("\nDONE")
if __name__ == "__main__":
    main()
