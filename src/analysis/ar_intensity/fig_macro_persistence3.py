"""Three-panel persistence CDF by prefix group: L0->L8, L8->L15, L0->L15. Offline, numpy-only."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/concept_ivt"; MINF = 400
PANELS = [("macro_persistence_L0L8.npz","L0 to L8"), ("macro_persistence.npz","L8 to L15"),
          ("macro_persistence_L0L15.npz","L0 to L15")]
groups = [("G0",0,256,"#c0392b"),("G1",256,512,"#e67e22"),("G2",512,1024,"#f1c40f"),
          ("G3",1024,2048,"#27ae60"),("G4",2048,4096,"#2980b9")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax,(fn,lab) in zip(axes, PANELS):
    try: d = np.load(f"{D}/{fn}")
    except FileNotFoundError: ax.set_title(f"{lab} (pending)"); continue
    both,c8,c15 = d["both"],d["cnt8"],d["cnt15"]
    best = (both/np.maximum(c8[:,None]+c15[None,:]-both,1)).max(1)
    meds = {}
    for n,a,b,col in groups:
        m = c8[a:b] >= MINF; v = np.sort(best[a:b][m])
        ax.plot(v, np.linspace(0,1,len(v)), color=col, label=n); meds[n] = round(float(np.median(v)),3)
    ax.set_title(lab); ax.set_xlabel("best source-to-target firing Jaccard"); ax.grid(alpha=.3)
    print(lab, "medians:", meds)
axes[0].set_ylabel("CDF over concepts"); axes[0].legend(fontsize=8, title="prefix group")
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/macro_persistence3.png", dpi=170, bbox_inches="tight")
print("saved macro_persistence3.png")
