"""Publication figure: clamping L15/L8 AR concepts during the Nov 2021 BC atmospheric river."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
D = "/scratch/euh7ys/climate_xai/patching"
def piv(f):
    df = pd.read_csv(f"{D}/{f}"); df["t"] = pd.to_datetime(df["time"])
    return df.pivot(index="t", columns="cond", values="ivt_max").sort_index()
l8 = piv("bc_ar_L8.csv"); l15S = piv("bc_ar_set.csv"); l15s = piv("bc_ar_2251.csv")
t = l8.index; base = l15S["baseline"]
d15_set = l15S["clamp_set"] - l15S["baseline"]
d8_set  = l8["clamp_set"]  - l8["baseline"]
noise = pd.DataFrame({"c15": l15S["control_set"] - l15S["baseline"],
                      "c8":  l8["control_set"]   - l8["baseline"],
                      "s15": l15s["clamp"]       - l15s["baseline"],
                      "s8":  l8["clamp_single"]  - l8["baseline"]})
nlo, nhi = noise.min(1), noise.max(1)
p0, p1 = pd.Timestamp("2021-11-14T00:00"), pd.Timestamp("2021-11-15T12:00")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.08))
ax1.fill_between(t, base, color="#cfe0f3"); ax1.plot(t, base, color="#23527c", lw=2)
ax1.axvspan(p0, p1, color="#f3e2c0", alpha=0.45, zorder=0)
pk = base.idxmax(); ax1.scatter([pk], [base.max()], color="#23527c", zorder=5)
ax1.annotate(f"AR peak  {base.max():.0f}", xy=(pk, base.max()), xytext=(6, -4),
             textcoords="offset points", fontsize=9, va="top")
ax1.set_ylabel("Baseline predicted\nregion max IVT\n(kg m$^{-1}$ s$^{-1}$)", fontsize=10)
ax1.set_title("Clamping Layer-15 / Layer-8 AR concepts during the Nov 2021 BC atmospheric river", fontsize=12)
ax1.margins(x=0.01)

ax2.axvspan(p0, p1, color="#f3e2c0", alpha=0.45, zorder=0)
ax2.axhline(0, color="#555", lw=0.8)
ax2.fill_between(t, nlo, nhi, color="#cccccc", alpha=0.7, label="control + single-concept (noise floor)")
ax2.plot(t, d15_set, "-o", color="#1b7837", lw=2.2, ms=5, label="clamp top-10 AR set, L15")
ax2.plot(t, d8_set,  "-s", color="#d1701a", lw=2.2, ms=5, label="clamp top-10 AR set, L8")
ax2.set_ylabel("Change in predicted IVT\n(clamp minus baseline,\nkg m$^{-1}$ s$^{-1}$)", fontsize=10)
ax2.set_xlabel("2021", fontsize=10)
ax2.legend(fontsize=9, loc="lower left", framealpha=0.9)
ax2.xaxis.set_major_formatter(DateFormatter("%b %d %Hz")); ax2.margins(x=0.01)
fig.autofmt_xdate(rotation=35)
out = "/scratch/euh7ys/climate_xai/plots/clamp_bc_ar.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)
