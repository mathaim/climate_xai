"""Dose-response figure: core IVT vs delta scale for child 3481 and parent 340 (spatially-specific causal dial)."""
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv("/scratch/euh7ys/climate_xai/patching/wsa_dose.csv")
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for cc, col, lab in [(3481, "#d62728", "child 3481 (S-Chile landfall) - moisture dial"),
                     (340, "#1f77b4", "parent 340 (storm track) - opposite sign")]:
    d = df[df.concept == cc].sort_values("scale")
    ax.plot(d.scale, d.core, "-o", color=col, label=lab)
base = df[(df.concept == 3481) & (df.scale == 0)].core.iloc[0]
ax.axvline(0, color="0.7", ls=":"); ax.axhline(base, color="0.85", ls="--", lw=0.8)
ax.set_xlabel("delta scale   (-1 amplify   0 baseline   +1 clamp   +2 over-clamp)")
ax.set_ylabel("predicted core IVT at the landfall point")
ax.set_title("Causal dose-response at the S-Chile AR landfall (2017-08-21)")
ax.text(0.02, 0.03, "regional box_max stays flat ~915 across all scales (spatial specificity)",
        transform=ax.transAxes, fontsize=8, color="0.4")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/wsa_dose.png", dpi=170, bbox_inches="tight"); print("saved")
