"""Fan-out WITHOUT normalization: raw mean activation vs IVT, shared y so magnitudes are
honestly comparable. Parent 99 large; rare children tiny on average (see firing fingerprint)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; REGN = list(REGIONS)
ROWS = [99, 3153, 3483]
ROLE = {99:"parent 99\n(general intensity)", 3153:"child 3153\n(extreme E. Aus)", 3483:"child 3483\n(N. Hemisphere)"}
NBIN = 10
def tuning(act, ivt):
    q = np.quantile(ivt, np.linspace(0,1,NBIN+1)); xc=[]; ym=[]
    for i in range(NBIN):
        m = (ivt>=q[i])&(ivt<=q[i+1])
        if m.sum(): xc.append((q[i]+q[i+1])/2); ym.append(act[m].mean())
    return np.array(xc), np.array(ym)
def main():
    data = {}
    for r in REGN:
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); ivt = t["ivt"].astype(float); ok = np.isfinite(ivt)
        data[r] = (ivt[ok], {c: t["A_max"][ok, c].astype(float) for c in ROWS}); del t
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(16, 9), sharey=True)
    for ri, cc in enumerate(ROWS):
        pk = max(data[r][1][cc].max() for r in REGN)
        for ci, r in enumerate(REGN):
            ivt, acts = data[r]; x, y = tuning(acts[cc], ivt)
            ax = axes[ri][ci]; ax.plot(x, y, "-o", ms=3, color="#c0392b" if ri==0 else "#2b6cb0")
            if ri==0: ax.set_title(r.replace("_"," "))
            if ci==0: ax.set_ylabel(f"c{cc} {ROLE[cc]}\nmean activation", fontsize=8)
            if ri==len(ROWS)-1: ax.set_xlabel("IVT")
        axes[ri][0].annotate(f"peak bin mean = {pk:.3f}", xy=(0.04,0.88), xycoords="axes fraction", fontsize=8)
    fig.suptitle("Fan-out, raw (un-normalized), shared y-axis", y=1.0)
    fig.tight_layout(); fig.savefig(f"{TRACK}/fanout_99_raw.png", dpi=150, bbox_inches="tight")
    print("saved", f"{TRACK}/fanout_99_raw.png")
if __name__ == "__main__":
    main()
