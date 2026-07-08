"""Architecture comparison: persistence CDF matry vs plain per transition, plus median best-J
by firing-rate quartile for both archs (the breadth control). Offline, numpy-only."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/concept_ivt"; MINF = 400
TR = [("L0L8","L0 to L8","macro_persistence_L0L8.npz","macro_persistence_plain_L0L8.npz"),
      ("L8L15","L8 to L15","macro_persistence.npz","macro_persistence_plain_L8L15.npz"),
      ("L0L15","L0 to L15","macro_persistence_L0L15.npz","macro_persistence_plain_L0L15.npz")]
def best_and_rate(fn):
    d = np.load(f"{D}/{fn}"); both,c8,c15 = d["both"],d["cnt8"],d["cnt15"]
    return (both/np.maximum(c8[:,None]+c15[None,:]-both,1)).max(1), c8
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax,(key,lab,fm,fp) in zip(axes, TR):
    for fn, col, name, ls in [(fm,"#8e44ad","Matryoshka","-"),(fp,"#16a085","Plain","--")]:
        try: best, c8 = best_and_rate(fn)
        except FileNotFoundError: continue
        m = c8 >= MINF; v = np.sort(best[m])
        ax.plot(v, np.linspace(0,1,len(v)), ls, color=col, label=name)
        q = np.quantile(c8[m],[0,.25,.5,.75,1.]); rb = np.clip(np.digitize(c8[m],q)-1,0,3)
        meds = [round(float(np.median(best[m][rb==r])),3) for r in range(4)]
        print(f"{lab:>10} {name:>10}: median best-J by firing-rate quartile (low->high): {meds}")
    ax.set_title(lab); ax.set_xlabel("best source-to-target firing Jaccard"); ax.grid(alpha=.3)
axes[0].set_ylabel("CDF over concepts"); axes[0].legend(fontsize=9)
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/arch_persistence.png", dpi=170, bbox_inches="tight")
print("saved arch_persistence.png")
