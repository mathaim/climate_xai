"""6-hourly zoom on the Dec 1995 W.N.America AR event: region IVT vs concept 1592 activation."""
import numpy as np, pandas as pd, datetime as DT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.analysis.ar_intensity.regions import index_to_datetime
C = 1592; D = "/scratch/euh7ys/climate_xai/concept_ivt"
d = np.load(f"{D}/track_pool_W_N_America.npz")
A = d["A_max"][:, C].astype(float); ivt = d["ivt"].astype(float); ti = d["tindex"]
dt = np.array([index_to_datetime(int(t)) for t in ti]); o = np.argsort(dt); dt, A, ivt = dt[o], A[o], ivt[o]
lo, hi = DT.datetime(1995, 11, 1), DT.datetime(1996, 1, 15)
m = (dt >= lo) & (dt <= hi); dt, A, ivt = dt[m], A[m], ivt[m]      # raw 6-hourly, no averaging
fig, axT = plt.subplots(figsize=(13, 4.6))
axT.plot(dt, ivt, color="#9aa0a6", lw=1.4)
axT.set_ylabel("region IVT\n(kg m$^{-1}$ s$^{-1}$)", color="#5f6368", fontsize=11); axT.tick_params(axis="y", labelcolor="#5f6368")
ax2 = axT.twinx(); ax2.plot(dt, A, color="#1b7837", lw=1.4)
ax2.set_ylabel("concept 1592 activation", color="#1b7837", fontsize=11); ax2.tick_params(axis="y", labelcolor="#1b7837")
axT.axvline(DT.datetime(1995, 12, 12, 6), color="#c0392b", ls="--", lw=1.2)
axT.set_xlim(lo, hi); axT.margins(x=0.01)
axT.xaxis.set_major_locator(mdates.DayLocator(interval=10)); axT.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
for lab in axT.get_xticklabels(): lab.set_rotation(30); lab.set_ha("right")
out = "/scratch/euh7ys/climate_xai/plots/zoom_1592_1995.png"
fig.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out)
