"""Final persistence figure: per transition, Matryoshka G0-G4 CDFs (colored) + Plain SAE as a
single dashed reference curve. Offline, numpy-only. No in-image title beyond panel tags."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/concept_ivt"; MINF = 400
TR = [("L0 to L8",  "macro_persistence_L0L8.npz",  "macro_persistence_plain_L0L8.npz"),
      ("L8 to L15", "macro_persistence.npz",       "macro_persistence_plain_L8L15.npz"),
      ("L0 to L15", "macro_persistence_L0L15.npz", "macro_persistence_plain_L0L15.npz")]
G = [("G0",0,256,"#c0392b"),("G1",256,512,"#e67e22"),("G2",512,1024,"#f1c40f"),
     ("G3",1024,2048,"#27ae60"),("G4",2048,4096,"#2980b9")]
def best(fn):
    d = np.load(f"{D}/{fn}"); b,c8,c15 = d["both"],d["cnt8"],d["cnt15"]
    return (b/np.maximum(c8[:,None]+c15[None,:]-b,1)).max(1), c8
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax,(lab,fm,fp) in zip(axes, TR):
    try:
        bm, c8 = best(fm)
        for n,a,b,col in G:
            m = c8[a:b] >= MINF; v = np.sort(bm[a:b][m])
            ax.plot(v, np.linspace(0,1,len(v)), color=col, lw=1.6, label=f"Matryoshka {n}")
    except FileNotFoundError: pass
    try:
        bp, cp = best(fp); m = cp >= MINF; v = np.sort(bp[m])
        ax.plot(v, np.linspace(0,1,len(v)), "--", color="0.25", lw=1.8, label="Plain SAE (all)")
    except FileNotFoundError: pass
    ax.set_title(lab); ax.set_xlabel("best source-to-target firing Jaccard")
    ax.set_xlim(0, 0.8); ax.grid(alpha=.3)
axes[0].set_ylabel("CDF over concepts"); axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/persistence_final.png", dpi=170, bbox_inches="tight")
print("saved persistence_final.png")
