"""Concept 1592 (Plain L8, W_N_America) activation vs region IVT, 1985-2013, with 3 AR snapshots."""
import numpy as np, datetime as DT, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.analysis.ar_intensity.regions import index_to_datetime
C = 1592
D = "/scratch/euh7ys/climate_xai/concept_ivt"
d = np.load(f"{D}/track_pool_W_N_America.npz")
A = d["A_max"][:, C].astype(float); ivt = d["ivt"].astype(float); tindex = d["tindex"]
dt = np.array([index_to_datetime(int(t)) for t in tindex])
o = np.argsort(dt); dt, A, ivt = dt[o], A[o], ivt[o]
lo, hi = DT.datetime(1985, 1, 1), DT.datetime(2013, 12, 31)
m = (dt >= lo) & (dt <= hi); dt, A, ivt = dt[m], A[m], ivt[m]
snaps = [DT.datetime(1986, 1, 12, 12), DT.datetime(2000, 11, 23, 0), DT.datetime(2012, 11, 28, 6)]
print("points", len(dt), "range", dt.min(), dt.max())
for s in snaps:
    i = np.argmin(np.abs(dt - s)); print("snap", s, "-> matched", dt[i], "A1592", round(A[i], 2), "IVT", round(ivt[i], 1))

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.38, wspace=0.22)
axT = fig.add_subplot(gs[0, :])
axT.plot(dt, ivt, color="#9aa0a6", lw=0.25, alpha=0.7)
axT.set_ylabel("region max IVT\n(kg m$^{-1}$ s$^{-1}$)", color="#5f6368", fontsize=10)
ax2 = axT.twinx()
ax2.plot(dt, A, color="#1b7837", lw=0.25, alpha=0.75)
ax2.set_ylabel("concept 1592 activation", color="#1b7837", fontsize=10)
for s in snaps:
    axT.axvline(s, color="#c0392b", lw=1.1, ls="--")
    axT.annotate(f"{s:%Y-%m-%d}", xy=(s, axT.get_ylim()[1]), xytext=(0, 2),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#c0392b")
axT.set_title("Concept 1592 (Plain L8, W. North America) tracks atmospheric-river IVT, 1985-2013", fontsize=12)
axT.set_xlim(lo, hi); axT.xaxis.set_major_locator(mdates.YearLocator(5))
axT.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

for k, s in enumerate(snaps):
    ax = fig.add_subplot(gs[1, k])
    w = (dt >= s - DT.timedelta(days=12)) & (dt <= s + DT.timedelta(days=12))
    tt, aa, vv = dt[w], A[w], ivt[w]
    nv = vv / (np.nanmax(vv) + 1e-9); na = aa / (np.nanmax(aa) + 1e-9)
    ax.plot(tt, nv, color="#9aa0a6", lw=1.4, label="IVT")
    ax.plot(tt, na, color="#1b7837", lw=1.4, label="1592")
    ax.axvline(s, color="#c0392b", lw=1.2, ls="--")
    ax.set_title(f"snapshot {s:%Y-%m-%d %Hz}", fontsize=9.5)
    ax.set_ylim(0, 1.08); ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for lab in ax.get_xticklabels(): lab.set_rotation(30); lab.set_ha("right")
    if k == 0:
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("normalized\n(0-1 per window)", fontsize=8)
out = "/scratch/euh7ys/climate_xai/plots/timeseries_1592_1985_2013.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
