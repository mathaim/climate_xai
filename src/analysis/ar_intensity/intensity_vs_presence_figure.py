"""Concept 99 (intensity: rises with IVT in all regions) vs concept 96 (presence: flat).
Activation binned by within-region IVT percentile, one line per region."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"
REG = list(REGIONS)
COLORS = {"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
PANELS = [(99,"Concept 99  (intensity)"),(96,"Concept 96  (presence)")]
NBIN = 12
def tuning(act, ivt):
    pct = ivt.argsort().argsort() / (len(ivt)-1) * 100
    e = np.linspace(0,100,NBIN+1); xc=[]; ym=[]
    for i in range(NBIN):
        m = (pct>=e[i])&(pct<=e[i+1])
        if m.sum(): xc.append((e[i]+e[i+1])/2); ym.append(act[m].mean())
    return np.array(xc), np.array(ym)
def main():
    data = {}
    for r in REG:
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); ivt = t["ivt"].astype(float); ok = np.isfinite(ivt)
        data[r] = (ivt[ok], {c: t["A_max"][ok, c].astype(float) for c,_ in PANELS}); del t
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (c, title) in zip(axes, PANELS):
        gmax = max(data[r][1][c].max() for r in REG) or 1
        for r in REG:
            ivt, acts = data[r]; x, y = tuning(acts[c], ivt)
            ax.plot(x, y/gmax, "-o", ms=3, color=COLORS[r], label=r.replace("_"," "))
        ax.set_title(title); ax.set_xlabel("Within-region IVT percentile"); ax.set_ylim(0,1.05); ax.grid(alpha=.3)
    axes[0].set_ylabel("Normalized activation"); axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{TRACK}/intensity_vs_presence.png", dpi=160, bbox_inches="tight")
    print("saved", f"{TRACK}/intensity_vs_presence.png")
if __name__ == "__main__":
    main()
