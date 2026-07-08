"""Offline plot from macro_persistence.npz: CDF of best-match L15 Jaccard for all L8 concepts,
split by Matryoshka prefix group; case-study concepts annotated. No in-image title."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("/scratch/euh7ys/climate_xai/concept_ivt/macro_persistence.npz")
both, cnt8, cnt15 = d["both"], d["cnt8"], d["cnt15"]; MINF = 400
J = both / np.maximum(cnt8[:,None] + cnt15[None,:] - both, 1); best = J.max(1)
groups = [("G0", 0, 256, "#c0392b"), ("G1", 256, 512, "#e67e22"), ("G2", 512, 1024, "#f1c40f"),
          ("G3", 1024, 2048, "#27ae60"), ("G4", 2048, 4096, "#2980b9")]
fig, ax = plt.subplots(figsize=(8, 5)); kept = 0
for name, a, b, col in groups:
    m = cnt8[a:b] >= MINF; v = np.sort(best[a:b][m]); kept += m.sum()
    ax.plot(v, np.linspace(0, 1, len(v)), color=col, label=f"{name} (n={m.sum()})")
for cc, lab in [(3481,"3481"), (99,"99"), (3392,"3392"), (340,"340")]:
    if cnt8[cc] >= MINF: ax.axvline(best[cc], color="0.4", ls=":", lw=0.8); ax.text(best[cc], 1.02, lab, fontsize=7, ha="center")
ax.set_xlabel("best L8$\\to$L15 firing Jaccard"); ax.set_ylabel("CDF over concepts")
ax.legend(fontsize=8, title="prefix group"); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/macro_persistence_cdf.png", dpi=170, bbox_inches="tight")
print(f"saved macro_persistence_cdf.png ({kept} concepts with >= {MINF} firings; nsteps={int(d['nsteps'])})")
med = lambda a,b: np.median(best[a:b][cnt8[a:b]>=MINF])
print("median best-J by group:", {n: round(float(med(a,b)),3) for n,a,b,_ in groups})
