"""Chance-normalized persistence control: per concept, best-match Jaccard divided by the 99th
percentile of its own 4096 overlaps (its chance floor, which is high for broad concepts). CDFs
by prefix group + Plain reference, per transition. Log-x. Offline, numpy-only."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/concept_ivt"; MINF = 400
TR = [("L0 to L8",  "macro_persistence_L0L8.npz",  "macro_persistence_plain_L0L8.npz"),
      ("L8 to L15", "macro_persistence.npz",       "macro_persistence_plain_L8L15.npz"),
      ("L0 to L15", "macro_persistence_L0L15.npz", "macro_persistence_plain_L0L15.npz")]
G = [("G0",0,256,"#c0392b"),("G1",256,512,"#e67e22"),("G2",512,1024,"#f1c40f"),
     ("G3",1024,2048,"#27ae60"),("G4",2048,4096,"#2980b9")]
def ratio(fn):
    d = np.load(f"{D}/{fn}"); b,c8,c15 = d["both"],d["cnt8"],d["cnt15"]
    J = b/np.maximum(c8[:,None]+c15[None,:]-b,1)
    p99 = np.percentile(J, 99, axis=1)
    return J.max(1)/np.maximum(p99, 1e-6), c8
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax,(lab,fm,fp) in zip(axes, TR):
    meds = {}
    try:
        r, c8 = ratio(fm)
        for n,a,b,col in G:
            m = c8[a:b] >= MINF; v = np.sort(r[a:b][m])
            ax.plot(v, np.linspace(0,1,len(v)), color=col, lw=1.6, label=f"Matryoshka {n}")
            meds[n] = round(float(np.median(v)),1)
    except FileNotFoundError: pass
    try:
        rp, cp = ratio(fp); m = cp >= MINF; v = np.sort(rp[m])
        ax.plot(v, np.linspace(0,1,len(v)), "--", color="0.25", lw=1.8, label="Plain SAE (all)")
        meds["plain"] = round(float(np.median(v)),1)
    except FileNotFoundError: pass
    ax.axvline(1.0, color="0.6", ls=":", lw=1)   # ratio 1 = at the chance floor
    ax.set_xscale("log"); ax.set_title(lab); ax.set_xlabel("best Jaccard / own 99th-pct background")
    ax.grid(alpha=.3, which="both")
    if meds: print(lab, "median best/background:", meds)
axes[0].set_ylabel("CDF over concepts"); axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/persistence_control.png", dpi=170, bbox_inches="tight")
print("saved persistence_control.png")
