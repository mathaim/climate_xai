"""Concept per-node peak activation vs region max-IVT, one wet-season window per region.
Reads cached track_pool (A_max) - no SAE encoding. Shared axes; season + full-record r/RMSE per panel."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import datetime
from src.analysis.ar_intensity.regions import index_to_datetime
TR="/scratch/euh7ys/climate_xai/concept_ivt"; PLOTS="/scratch/euh7ys/climate_xai/plots"
CC={"W_N_America":1592,"W_Europe":2948,"W_S_America":3218,"E_Australia":3720}
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe",
      "W_S_America":"Western South America","E_Australia":"Eastern Australia"}
WIN={"W_N_America":(datetime(2015,12,1),datetime(2016,2,28)),
     "W_Europe":(datetime(2015,12,1),datetime(2016,2,28)),
     "W_S_America":(datetime(2015,6,1),datetime(2015,8,31)),
     "E_Australia":(datetime(2015,6,1),datetime(2015,8,31))}
REGIONS=["W_N_America","W_Europe","W_S_America","E_Australia"]
def seaslabel(lo,hi):
    return f"winter {lo.year}/{hi.year}" if lo.year!=hi.year else f"winter {lo.year}"
def seaslabel(lo,hi):
    return f"winter {lo.year}/{hi.year}" if lo.year!=hi.year else f"winter {lo.year}"
def fit(x,y):
    m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    if len(x)<3: return float("nan"),float("nan")
    r=float(np.corrcoef(x,y)[0,1]); sl,ic=np.polyfit(x,y,1)
    return r, float(np.sqrt(np.mean((y-(sl*x+ic))**2)))
fig,axes=plt.subplots(4,1,figsize=(15,15))
for ax,reg in zip(axes,REGIONS):
    cc=CC[reg]; d=np.load(f"{TR}/track_pool_{reg}.npz")
    A=d["A_mean"][:,cc].astype(float); IV=d["ivt"].astype(float); ti=d["tindex"]
    scale=float(np.nanpercentile(A[np.isfinite(A)],99.5))  # per-concept normalizer (does not change r)
    dts=np.array([index_to_datetime(int(t)) for t in ti])
    lo,hi=WIN[reg]; mw=(dts>=lo)&(dts<=hi); o=np.argsort(dts[mw])
    T=dts[mw][o]; Aw=A[mw][o]; IVw=IV[mw][o]
    rw,rmw=fit(Aw,IVw); rf,rmf=fit(A,IV)
    Aw=Aw/scale
    ax.plot(T,Aw,color="#c0392b",lw=1.5)
    ax2=ax.twinx(); ax2.plot(T,IVw,color="#185FA5",lw=1.5,alpha=.8)
    ax.set_ylim(-0.04,1.20); ax2.set_ylim(60,1260)
    ax.set_ylabel("normalized activation",color="#c0392b",fontsize=15)
    ax2.set_ylabel("max IVT (kg m$^{-1}$ s$^{-1}$)",color="#185FA5",fontsize=15)
    ax.tick_params(axis="y",labelsize=12,colors="#c0392b"); ax2.tick_params(axis="y",labelsize=12,colors="#185FA5")
    ax.tick_params(axis="x",labelsize=12)
    ax.set_title(f"Concept {cc} vs. {NAME[reg]} IVT",fontsize=20,loc="center",fontweight="bold",pad=28)
    ax.grid(alpha=.3)
    ax.text(0.5,1.012,
            f"{seaslabel(lo,hi)}: r={rw:.2f}, RMSE={rmw:.0f}     |     1979-2017: r={rf:.2f}, RMSE={rmf:.0f}",
            transform=ax.transAxes,va="bottom",ha="center",fontsize=12)
    print(f"{reg:13} c{cc} n_win={len(T)} season r={rw:.2f} RMSE={rmw:.0f} | full r={rf:.2f} RMSE={rmf:.0f}")
fig.tight_layout()
fig.savefig(f"{PLOTS}/timeseries_concept_ivt.png",dpi=160,bbox_inches="tight")
print("saved", f"{PLOTS}/timeseries_concept_ivt.png")
