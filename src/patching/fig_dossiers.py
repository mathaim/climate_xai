"""Intervention dossiers: per concept, the causal cascade vs dose g (0 clamp, 1 base, 2-3 amp).
Latent rows as % of baseline firing; output row in physical IVT units."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = [0, 1, 2, 3]
def pct(vals): b = vals[1]; return [100.0*(v-b)/b for v in vals]
def panel(ax, series, cols, title, ylab="% change vs baseline"):
    for lab, v in series.items():
        ax.plot(G[:len(v)] if len(v)==4 else [1,2,3], v, "-o", ms=4, color=cols[lab], label=lab)
    ax.axhline(0, color="0.75", lw=0.8); ax.axvline(1, color="0.9", lw=0.8, ls=":")
    ax.set_ylabel(ylab, fontsize=8); ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25)
    ax.set_xticks(G); ax.set_xticklabels(["clamp\ng=0","baseline\ng=1","amplify\ng=2","amplify\ng=3"], fontsize=7)

# ---------------- 99 dossier ----------------
fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
panel(ax[0], {"1454": pct([4.2,5.8,7.6,10.0]), "3392": pct([3.1,4.5,7.6,12.6]), "2722": pct([11.2,13.7,18.1,24.3])},
      {"1454":"#8e44ad","3392":"#e67e22","2722":"#d81b60"}, "(a) layer-8 children of 99")
panel(ax[1], {"111": pct([130.2-76.0,130.2,130.2+57.9,130.2+82.0]), "214": pct([70.1-26.4,70.1,70.1+43.3,70.1+92.5]),
              "123": pct([86.6-25.9,86.6,86.6+12.0,86.6+16.1]), "864": pct([13.7-6.2,13.7,13.7+10.0,13.7+17.4])},
      {"111":"#c0392b","214":"#e67e22","123":"#f1c40f","864":"#7f8c8d"}, "(b) layer-15 committee and absorber")
panel(ax[2], {"1269": pct([8.7-2.8,8.7,8.7+11.1,8.7+23.3]), "2739": pct([7.1-4.4,7.1,7.1+16.7,7.1+27.0])},
      {"1269":"#2980b9","2739":"#27ae60"}, "(c) children of the layer-15 core (111)")
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/dossier_99.png", dpi=170, bbox_inches="tight")
print("saved dossier_99.png")

# ---------------- 340 dossier ----------------
fig, ax = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
panel(ax[0], {"3481": pct([0.9,1.2,1.4,1.9]), "3948": pct([1.1,1.2,1.4,1.6]), "3675": pct([0.7,0.9,1.0,1.2])},
      {"3481":"#d62728","3948":"#ff7f0e","3675":"#2ca02c"}, "(a) layer-8 children of 340")
panel(ax[1], {"1536": pct([3.5,3.5,3.5+3.8,3.5+7.6])[1:], "1675": pct([8.2,8.2,8.2+1.2,8.2+2.4])[1:],
              "3160": pct([1.1,1.1,1.1+0.6,1.1+1.3])[1:]},
      {"1536":"#2980b9","1675":"#7f8c8d","3160":"#d62728"}, "(b) layer-15 fragments (amplification only)")
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/dossier_340.png", dpi=170, bbox_inches="tight")
print("saved dossier_340.png")
