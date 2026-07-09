"""Org map per region: same 2x3 (arch x layer) layout, but corr computed against ONE region's
AR intensity. Solid circle = that region's champion; dashed circle = the global (4-region avg)
champion, to test whether the core is region-general. Caches per-region corr vectors."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
C = "/scratch/euh7ys/climate_xai/concept_ivt"; OUT = "/scratch/euh7ys/climate_xai/plots"
REGIONS = ["W_N_America", "W_S_America", "W_Europe", "E_Australia"]
GC = {"G0": "#c0392b", "G1": "#e67e22", "G2": "#f1c40f", "G3": "#27ae60", "G4": "#2980b9"}
GRP = lambda c: "G0" if c < 256 else "G1" if c < 512 else "G2" if c < 1024 else "G3" if c < 2048 else "G4"
TOT = 8000 * 40962.0
CNT = {("matry","L0"): ("macro_persistence_L0L8.npz","cnt8"), ("matry","L8"): ("macro_persistence.npz","cnt8"),
       ("matry","L15"): ("macro_persistence.npz","cnt15"), ("plain","L0"): ("macro_persistence_plain_L0L8.npz","cnt8"),
       ("plain","L8"): ("macro_persistence_plain_L8L15.npz","cnt8"), ("plain","L15"): ("macro_persistence_plain_L8L15.npz","cnt15")}
def region_corrs(sae):
    if all(np.os.path.exists(f"{C}/ar_corr_{sae}_{r}.npy") for r in REGIONS) if hasattr(np, "os") else False: pass
    import os
    missing = [r for r in REGIONS if not os.path.exists(f"{C}/ar_corr_{sae}_{r}.npy")]
    if missing:
        meta = pd.read_parquet(f"{D}/{sae}_meta.parquet"); F = np.load(f"{D}/{sae}_features_region_binary.npy", mmap_mode="r")
        iv_all = meta["max_ivt"].values.astype(float)
        for r in missing:
            m = (meta["region"] == r).values & np.isfinite(iv_all)
            X = np.asarray(F[m], dtype=np.float64); iv = iv_all[m]
            Xz = X - X.mean(0); ivz = iv - iv.mean(); den = np.sqrt((Xz**2).sum(0) * (ivz**2).sum())
            np.save(f"{C}/ar_corr_{sae}_{r}.npy", np.where(den > 0, (Xz*ivz[:,None]).sum(0)/np.maximum(den,1e-12), 0.0))
    return {r: np.load(f"{C}/ar_corr_{sae}_{r}.npy") for r in REGIONS}
RC = {f"{a}_{L}": region_corrs(f"{a}_{L}") for a in ["matry","plain"] for L in ["L0","L8","L15"]}
GA = {k: np.load(f"{C}/ar_corr_{k}.npy") for k in RC}   # global 4-region average (cached earlier)
for reg in REGIONS:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for r_i, arch in enumerate(["matry", "plain"]):
        for k, L in enumerate(["L0", "L8", "L15"]):
            ax = axes[r_i, k]; sae = f"{arch}_{L}"
            corr = RC[sae][reg]; fn, key = CNT[(arch, L)]; rate = np.load(f"{C}/{fn}")[key] / TOT
            m = rate > 1e-7
            cols = [GC[GRP(i)] for i in np.arange(4096)[m]] if arch == "matry" else "0.55"
            ax.scatter(rate[m], corr[m], s=5, c=cols, alpha=0.45, edgecolor="none")
            ch = int(np.argmax(corr)); gc = int(np.argmax(GA[sae]))
            ax.scatter([rate[ch]], [corr[ch]], s=90, facecolor="none", edgecolor="k", lw=1.4, zorder=5)
            ax.annotate(str(ch), (rate[ch], corr[ch]), textcoords="offset points", xytext=(7, 5), fontsize=9, weight="bold", zorder=6)
            if gc != ch:
                ax.scatter([rate[gc]], [corr[gc]], s=95, facecolor="none", edgecolor="0.3", lw=1.2, ls="--", zorder=5)
                ax.annotate(f"{gc} (global)", (rate[gc], corr[gc]), textcoords="offset points", xytext=(7, -12), fontsize=7.5, color="0.3", zorder=6)
            ax.set_xscale("log"); ax.axhline(0, color="0.8", lw=0.8); ax.grid(alpha=.25)
            ax.tick_params(labelbottom=True); ax.set_ylim(-0.44, 0.72)
            if r_i == 0: ax.set_title(f"layer {L[1:]}", fontsize=12)
    for r_i, lab in enumerate(["Matryoshka SAE", "Plain SAE"]):
        axes[r_i, 0].set_ylabel(f"{lab}\ncorr with {reg} AR intensity")
    for k in range(3): axes[1, k].set_xlabel("concept breadth (firing rate, log)")
    h = [plt.Line2D([0],[0], marker="o", ls="", mfc=GC[g], mec="none", label=g) for g in GC]
    axes[0, 0].legend(handles=h, fontsize=8, title="prefix group", loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/org_map_{reg}.png", dpi=170, bbox_inches="tight")
    print(f"saved org_map_{reg}.png  champions:", {f"{a}_{L}": int(np.argmax(RC[f'{a}_{L}'][reg])) for a in ["matry","plain"] for L in ["L0","L8","L15"]})
