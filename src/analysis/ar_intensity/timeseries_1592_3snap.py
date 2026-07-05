"""Concept 1592 (Plain L8, W_N_America) activation vs region IVT, 1985-2013, 3 AR snapshots marked.
Single panel, every-other-month maxima. Title goes in the LaTeX caption, not the figure."""
import numpy as np, pandas as pd, datetime as DT, os
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
snaps = [DT.datetime(1986, 1, 13, 0), DT.datetime(1995, 12, 12, 6), DT.datetime(2006, 11, 19, 12)]
env = pd.DataFrame({"A": A, "ivt": ivt}, index=pd.DatetimeIndex(dt)).sort_index().resample("2MS").mean()
print("points:", len(env))

fig, axT = plt.subplots(figsize=(14, 4.6))
axT.plot(env.index, env["ivt"], color="#9aa0a6", lw=1.3)
axT.set_ylabel("region max IVT\n(kg m$^{-1}$ s$^{-1}$)", color="#5f6368", fontsize=11)
axT.tick_params(axis="y", labelcolor="#5f6368")
ax2 = axT.twinx()
ax2.plot(env.index, env["A"], color="#1b7837", lw=1.3)
ax2.set_ylabel("concept 1592 activation", color="#1b7837", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#1b7837")
for s in snaps:
    axT.axvline(s, color="#c0392b", lw=1.2, ls="--")
    axT.annotate(f"{s:%Y-%m-%d}", xy=(s, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, color="#c0392b")
axT.set_xlim(lo, hi); axT.margins(x=0.01)
axT.xaxis.set_major_locator(mdates.YearLocator(5)); axT.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
out = "/scratch/euh7ys/climate_xai/plots/timeseries_1592_1985_2013.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
