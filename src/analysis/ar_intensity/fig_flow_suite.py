"""Figures for the convergence/emergence/addressing findings. Computes per-layer AR corr from
pipeline features (cached to npz), reads macro co-firing npz for edges. Saves 3 PNGs."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
C = "/scratch/euh7ys/climate_xai/concept_ivt"; OUT = "/scratch/euh7ys/climate_xai/plots"
GC = {"G0": "#c0392b", "G1": "#e67e22", "G2": "#f1c40f", "G3": "#27ae60", "G4": "#2980b9"}
GRP = lambda c: "G0" if c < 256 else "G1" if c < 512 else "G2" if c < 1024 else "G3" if c < 2048 else "G4"
def corr_vec(sae):
    fp = f"{C}/ar_corr_{sae}.npy"
    try: return np.load(fp)
    except FileNotFoundError: pass
    meta = pd.read_parquet(f"{D}/{sae}_meta.parquet"); F = np.load(f"{D}/{sae}_features_region_binary.npy", mmap_mode="r")
    iv_all = meta["max_ivt"].values.astype(float); cs = []
    for reg in meta["region"].unique():
        m = (meta["region"] == reg).values & np.isfinite(iv_all)
        X = np.asarray(F[m], dtype=np.float64); iv = iv_all[m]
        Xz = X - X.mean(0); ivz = iv - iv.mean(); den = np.sqrt((Xz**2).sum(0) * (ivz**2).sum())
        cs.append(np.where(den > 0, (Xz*ivz[:,None]).sum(0)/np.maximum(den,1e-12), 0.0))
    c = np.nan_to_num(np.mean(cs, 0)); np.save(fp, c); return c
def jac(fn):
    d = np.load(f"{C}/{fn}"); b, c8, c15 = d["both"], d["cnt8"], d["cnt15"]
    return b / np.maximum(c8[:, None] + c15[None, :] - b, 1)
cm = {L: corr_vec(f"matry_{L}") for L in ["L0", "L8", "L15"]}
cp = {L: corr_vec(f"plain_{L}") for L in ["L0", "L8", "L15"]}
J08, J815 = jac("macro_persistence_L0L8.npz"), jac("macro_persistence.npz")

# ---- Fig 1: flow diagram ----
TOP = {L: list(np.argsort(-cm[L])[:6]) for L in cm}
pos = {}; fig, ax = plt.subplots(figsize=(10, 6))
for xi, L in enumerate(["L0", "L8", "L15"]):
    for yi, cc in enumerate(TOP[L]):
        pos[(L, cc)] = (xi, -yi)
        g = GRP(cc); ax.scatter(xi, -yi, s=900, c=GC[g], zorder=3, edgecolor="k", lw=0.6)
        ax.text(xi, -yi, str(cc), ha="center", va="center", fontsize=8, color="white", zorder=4, weight="bold")
        ax.text(xi, -yi-0.32, f"{g}  r={cm[L][cc]:+.2f}", ha="center", fontsize=6.5, color="0.3", zorder=4)
for (La, Lb, Jm) in [("L0", "L8", J08), ("L8", "L15", J815)]:
    for ca in TOP[La]:
        for cb in TOP[Lb]:
            j = Jm[ca, cb]
            if j >= 0.05:
                (x1, y1), (x2, y2) = pos[(La, ca)], pos[(Lb, cb)]
                ax.plot([x1, x2], [y1, y2], "-", color="0.55", lw=min(8*j, 4), alpha=0.7, zorder=1)
                ax.text((x1+x2)/2, (y1+y2)/2 + 0.09, f"{j:.2f}", fontsize=6, color="0.4", ha="center", zorder=2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["layer 0", "layer 8", "layer 15"], fontsize=11)
ax.set_yticks([]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(-len(TOP["L0"]) + 0.2, 0.9)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/ar_flow_graph.png", dpi=170, bbox_inches="tight"); print("saved ar_flow_graph.png")

# ---- Fig 2: same function, different addressing ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
xs = [0, 8, 15]
ax[0].plot(xs, [cm[L].max() for L in ["L0","L8","L15"]], "-o", color="#8e44ad", label="Matryoshka")
ax[0].plot(xs, [cp[L].max() for L in ["L0","L8","L15"]], "--s", color="#16a085", label="Plain")
ax[0].set_xticks(xs); ax[0].set_xlabel("processor layer"); ax[0].set_ylabel("max corr with AR intensity")
ax[0].set_ylim(0, 0.6); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].axhspan(0, 256, color="#c0392b", alpha=0.12)
ax[1].text(15.4, 128, "G0 core", fontsize=8, color="#c0392b", va="center")
for L, x in zip(["L0","L8","L15"], xs):
    tm, tp = np.argsort(-cm[L])[:10], np.argsort(-cp[L])[:10]
    ax[1].scatter([x-0.55]*10, tm, s=60*np.abs(cm[L][tm])/0.5+10, c=[GC[GRP(t)] for t in tm], edgecolor="k", lw=0.3, label="Matryoshka" if L=="L0" else None)
    ax[1].scatter([x+0.55]*10, tp, s=60*np.abs(cp[L][tp])/0.5+10, c="0.6", marker="s", edgecolor="k", lw=0.3, label="Plain" if L=="L0" else None)
ax[1].set_xticks(xs); ax[1].set_xlabel("processor layer"); ax[1].set_ylabel("dictionary index of top-10 AR concepts")
ax[1].legend(fontsize=8, loc="upper left"); ax[1].grid(alpha=.2)
fig.tight_layout(); fig.savefig(f"{OUT}/ar_addressing.png", dpi=170, bbox_inches="tight"); print("saved ar_addressing.png")

# ---- Fig 3: convergence scatter at L15 ----
fig, ax = plt.subplots(figsize=(7, 5.5))
x, y = cm["L15"], J815[99]
cols = [GC[GRP(i)] for i in range(4096)]
ax.scatter(x, y, s=8, c=cols, alpha=0.45, edgecolor="none")
for cc in [111, 214, 123, 864]:
    ax.annotate(str(cc), (x[cc], y[cc]), textcoords="offset points", xytext=(6, 4), fontsize=9, weight="bold")
    ax.scatter([x[cc]], [y[cc]], s=60, facecolor="none", edgecolor="k", lw=1.2)
ax.set_xlabel("functional selection: corr with AR intensity (fresh, at layer 15)")
ax.set_ylabel("matching: firing Jaccard with layer-8 concept 99")
ax.grid(alpha=.3)
h = [plt.Line2D([0],[0], marker="o", ls="", mfc=GC[g], mec="none", label=g) for g in GC]
ax.legend(handles=h, fontsize=8, title="prefix group")
fig.tight_layout(); fig.savefig(f"{OUT}/ar_convergence.png", dpi=170, bbox_inches="tight"); print("saved ar_convergence.png")
