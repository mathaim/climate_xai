"""Intuitive version: predicted AR strength, normal vs with top-10 AR concepts deleted."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
D = "/scratch/euh7ys/climate_xai/patching"
def piv(f):
    df = pd.read_csv(f"{D}/{f}"); df["t"] = pd.to_datetime(df["time"])
    return df.pivot(index="t", columns="cond", values="ivt_max").sort_index()
l15S = piv("bc_ar_set.csv"); l8 = piv("bc_ar_L8.csv")
t = l15S.index; base = l15S["baseline"]; clamp = l15S["clamp_set"]; clamp8 = l8["clamp_set"]

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.plot(t, base,  color="#23527c", lw=2.6, label="normal forecast")
ax.plot(t, clamp, color="#d1701a", lw=2.2, label="forecast with the 10 strongest AR concepts deleted")
ax.fill_between(t, clamp, base, where=(base >= clamp), interpolate=True, color="#d1701a", alpha=0.18)
ax.set_ylabel("Predicted atmospheric-river strength over British Columbia\n"
              "(region max IVT, kg m$^{-1}$ s$^{-1}$)", fontsize=10)
ax.set_xlabel("2021", fontsize=10)
ax.set_title("Deleting GraphCast's AR concepts barely weakens the predicted Nov 2021 BC storm", fontsize=12)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax.xaxis.set_major_formatter(DateFormatter("%b %d %Hz")); ax.margins(x=0.01)

# zoom inset on the 2-day peak so the small gap is visible
axin = ax.inset_axes([0.085, 0.50, 0.40, 0.45])
axin.plot(t, base,  color="#23527c", lw=2.6)
axin.plot(t, clamp, color="#d1701a", lw=2.2)
axin.fill_between(t, clamp, base, where=(base >= clamp), interpolate=True, color="#d1701a", alpha=0.18)
axin.set_xlim(pd.Timestamp("2021-11-14T00:00"), pd.Timestamp("2021-11-15T18:00"))
axin.set_ylim(850, 1120); axin.set_title("zoom on the peak", fontsize=9)
axin.xaxis.set_major_formatter(DateFormatter("%d %Hz")); axin.tick_params(labelsize=8)
ax.indicate_inset_zoom(axin, edgecolor="#888")

pk = (base - clamp).idxmax(); gap = (base - clamp).max()
ax.annotate(f"biggest reduction here:\nonly ~{gap:.0f} out of ~{base[pk]:.0f}  (about 1%)",
            xy=(pk, clamp[pk]), xytext=(0.52, 0.16), textcoords="axes fraction", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#555"))
fig.text(0.5, -0.02, "Deleting the same concepts at layer 8 instead of layer 15 gives a nearly identical curve.",
         ha="center", fontsize=8.5, style="italic", color="#555")
out = "/scratch/euh7ys/climate_xai/plots/clamp_bc_ar_curves.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)
