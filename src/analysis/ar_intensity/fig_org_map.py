"""The macro synthesis figure: concept organization across layers x architectures.
Each panel: all 4096 concepts as (firing rate, AR-intensity corr), colored by prefix group
(matryoshka) or grey (plain). Champions annotated. All inputs cached; no encoding."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = "/scratch/euh7ys/climate_xai/concept_ivt"; TOT = 8000 * 40962.0
GC = {"G0": "#c0392b", "G1": "#e67e22", "G2": "#f1c40f", "G3": "#27ae60", "G4": "#2980b9"}
GRP = lambda c: "G0" if c < 256 else "G1" if c < 512 else "G2" if c < 1024 else "G3" if c < 2048 else "G4"
CNT = {("matry","L0"): ("macro_persistence_L0L8.npz","cnt8"), ("matry","L8"): ("macro_persistence.npz","cnt8"),
       ("matry","L15"): ("macro_persistence.npz","cnt15"), ("plain","L0"): ("macro_persistence_plain_L0L8.npz","cnt8"),
       ("plain","L8"): ("macro_persistence_plain_L8L15.npz","cnt8"), ("plain","L15"): ("macro_persistence_plain_L8L15.npz","cnt15")}
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
plt.rcParams["axes.xmargin"] = 0.05
for r, arch in enumerate(["matry", "plain"]):
    for k, L in enumerate(["L0", "L8", "L15"]):
        ax = axes[r, k]
        corr = np.load(f"{C}/ar_corr_{arch}_{L}.npy")
        fn, key = CNT[(arch, L)]; rate = np.load(f"{C}/{fn}")[key] / TOT
        m = rate > 1e-7
        cols = [GC[GRP(i)] for i in np.arange(4096)[m]] if arch == "matry" else "0.55"
        ax.scatter(rate[m], corr[m], s=5, c=cols, alpha=0.45, edgecolor="none")
        ch = int(np.argmax(corr))
        ax.scatter([rate[ch]], [corr[ch]], s=90, facecolor="none", edgecolor="k", lw=1.4, zorder=5)
        ax.annotate(str(ch), (rate[ch], corr[ch]), textcoords="offset points", xytext=(7, 5), fontsize=9, weight="bold", zorder=6)
        ax.set_xscale("log"); ax.axhline(0, color="0.8", lw=0.8); ax.grid(alpha=.25)
        ax.tick_params(labelbottom=True)
        if r == 0: ax.set_title(f"layer {L[1:]}", fontsize=12)
for r, lab in enumerate(["Matryoshka SAE", "Plain SAE"]):
    axes[r, 0].set_ylabel(f"{lab}\ncorr with AR intensity")
for k in range(3): axes[1, k].set_xlabel("concept breadth (firing rate, log)")
h = [plt.Line2D([0],[0], marker="o", ls="", mfc=GC[g], mec="none", label=g) for g in GC]
axes[0, 0].legend(handles=h, fontsize=8, title="prefix group", loc="upper left")
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/org_map.png", dpi=170, bbox_inches="tight")
print("saved org_map.png")
