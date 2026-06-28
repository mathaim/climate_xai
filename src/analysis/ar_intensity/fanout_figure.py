"""Fan-out figure: a global parent and its region-specific children, as activation-vs-IVT
tuning curves across all 4 regions. Parent rises everywhere; each child only in its region."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"
GROUPS = [(96, [2831, 3105]), (99, [3153, 3483])]   # parent -> children
ROLE = {96: "global parent", 99: "global parent",
        2831: "child: W_N_America", 3105: "child: W_S_America",
        3153: "child: E_Australia", 3483: "child: W_N_America"}
NBIN = 10
def tuning(act, ivt):
    q = np.quantile(ivt, np.linspace(0, 1, NBIN + 1)); xc, ym = [], []
    for i in range(NBIN):
        m = (ivt >= q[i]) & (ivt <= q[i + 1])
        if m.sum(): xc.append((q[i] + q[i + 1]) / 2); ym.append(act[m].mean())
    return np.array(xc), np.array(ym)
def main():
    concepts = sorted({c for p, ch in GROUPS for c in [p] + ch})
    data = {}
    for r in REGIONS:
        d = np.load(f"{TRACK}/track_matry_{r}.npz"); ivt = d["ivt"].astype(float); ok = np.isfinite(ivt)
        A = d["A_max"]; data[r] = (ivt[ok], {c: A[ok, c].astype(float) for c in concepts}); del d
    for p, ch in GROUPS:
        rows = [p] + ch
        fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.1 * len(rows)), squeeze=False)
        for ri, cc in enumerate(rows):
            gmax = max((data[r][1][cc].max() for r in REGIONS), default=1) or 1
            for ci, r in enumerate(REGIONS):
                ivt, acts = data[r]; x, y = tuning(acts[cc], ivt)
                ax = axes[ri][ci]; ax.plot(x, y / gmax, "-o", ms=3,
                                           color="#c0392b" if ri == 0 else "#2b6cb0")
                ax.set_ylim(0, 1.05)
                if ri == 0: ax.set_title(r)
                if ci == 0: ax.set_ylabel(f"c{cc}\n{ROLE.get(cc,'')}\nnorm. activation", fontsize=9)
                if ri == len(rows) - 1: ax.set_xlabel("IVT")
        fig.suptitle(f"Matryoshka fan-out: parent {p} -> region-specific children", y=1.01)
        fig.tight_layout(); fig.savefig(f"{TRACK}/fanout_{p}.png", dpi=150, bbox_inches="tight")
        print("saved", f"{TRACK}/fanout_{p}.png")
    print("DONE")
if __name__ == "__main__":
    main()
