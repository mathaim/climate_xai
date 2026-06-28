"""Paper figure: un-normalized Matryoshka fan-out. Parent 99 (frequent, broad intensity)
vs children 3153/3483 (rare, strong, region-locked). Per panel: binned-mean activation,
peak single-event activation, and firing frequency (% of the region's ARs with act>0.1)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; REGN = list(REGIONS)
ROWS = [(99,  "Concept 99 (parent)\nGeneral intensity",      "#c0392b"),
        (3153,"Concept 3153 (child)\nExtreme E. Australia",   "#1f4e79"),
        (3483,"Concept 3483 (child)\nNorthern Hemisphere",    "#1f4e79")]
NBIN = 10; THRESH = 0.1; AR = 250.0
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12})
def tuning(act, ivt):
    q = np.quantile(ivt, np.linspace(0, 1, NBIN + 1)); xc = []; ym = []
    for i in range(NBIN):
        m = (ivt >= q[i]) & (ivt <= q[i + 1])
        if m.sum(): xc.append((q[i] + q[i + 1]) / 2); ym.append(act[m].mean())
    return np.array(xc), np.array(ym)
def main():
    cset = [c for c, _, _ in ROWS]; data = {}
    for r in REGN:
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); ivt = t["ivt"].astype(float); ok = np.isfinite(ivt)
        data[r] = (ivt[ok], {c: t["A_max"][ok, c].astype(float) for c in cset}); del t
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(16, 9.5), sharey=True)
    for ri, (cc, label, col) in enumerate(ROWS):
        for ci, r in enumerate(REGN):
            ivt, acts = data[r]; a = acts[cc]; x, y = tuning(a, ivt)
            ax = axes[ri][ci]; ax.plot(x, y, "-o", ms=3, lw=1.8, color=col)
            ax.grid(alpha=.25); ax.set_ylim(0, 0.30)
            ar = ivt >= AR
            peak = float(a.max()); frate = 100 * (a[ar] > THRESH).mean() if ar.any() else 0.0
            ax.text(0.05, 0.94, f"peak {peak:.2f}\nfires {frate:.0f}% of ARs",
                    transform=ax.transAxes, fontsize=9.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=.9))
            if ri == 0: ax.set_title(r.replace("_", " "))
            if ci == 0: ax.set_ylabel(label, fontsize=10.5)
            if ri == len(ROWS) - 1: ax.set_xlabel("IVT (kg m$^{-1}$ s$^{-1}$)")
    fig.supylabel("Mean concept activation", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{TRACK}/fanout_paper.png", dpi=200, bbox_inches="tight")
    print("saved", f"{TRACK}/fanout_paper.png")
if __name__ == "__main__":
    main()
