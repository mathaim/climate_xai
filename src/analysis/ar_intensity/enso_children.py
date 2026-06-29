"""Do concept 99's two children track ENSO oppositely? 3153 (E. Australia) vs 3483 (N. America / Europe).
DJF strong-firing count per season vs DJF Nino 3.4 (ONI)."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
ONI = {1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,
       1988:0.81,1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,
       1996:-0.90,1997:-0.50,1998:2.24,1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,
       2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,2009:-0.85,2010:1.50,2011:-1.31,
       2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
PCOL = {"La Nina":"#2b6cb0","Neutral":"#9e9e9e","El Nino":"#c0392b"}
def phase(o): return "La Nina" if o <= -0.5 else ("El Nino" if o >= 0.5 else "Neutral")
PAIRS = [(3153,"E_Australia","Child 3153  \u00b7  E. Australia"),
         (3483,"W_N_America","Child 3483  \u00b7  W. North America"),
         (3483,"W_Europe","Child 3483  \u00b7  W. Europe")]
def season_counts(c, region):
    t = np.load(f"{TRACK}/track_matry_{region}.npz")
    a = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float); ti = t["tindex"]
    ok = np.isfinite(ivt); a, ti = a[ok], ti[ok]
    d = AR_START + pd.to_timedelta(6*(ti-1), unit="h"); yr = d.year.values; mo = d.month.values
    syr = yr + (mo == 12); fire = a >= np.quantile(a, 0.99)
    out = []
    for y in sorted(set(syr.tolist())):
        if y not in ONI: continue
        m = (syr == y) & np.isin(mo, [12,1,2,3]); out.append((ONI[y], int((fire & m).sum())))
    return np.array(out)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (c, region, title) in zip(axes, PAIRS):
    R = season_counts(c, region); oni, cnt = R[:,0], R[:,1]; ph = [phase(o) for o in oni]
    r = float(np.corrcoef(cnt, oni)[0,1])
    ln = cnt[[p=="La Nina" for p in ph]].mean(); en = cnt[[p=="El Nino" for p in ph]].mean()
    print(f"{title}:  r(count, Nino3.4) = {r:+.2f}   La Nina mean = {ln:.1f}   El Nino mean = {en:.1f}")
    ax.scatter(oni, cnt, c=[PCOL[p] for p in ph], s=45, edgecolor="k", lw=.4)
    b, a0 = np.polyfit(oni, cnt, 1); xs = np.linspace(oni.min(), oni.max(), 10); ax.plot(xs, a0+b*xs, "k-", lw=1)
    ax.axvline(-0.5, ls="--", c="#2b6cb0", lw=.7); ax.axvline(0.5, ls="--", c="#c0392b", lw=.7)
    ax.set_title(f"{title}\nr = {r:+.2f}", fontsize=10.5); ax.set_xlabel("DJF Nino 3.4 (ONI)")
axes[0].set_ylabel("strong firings per DJF season")
axes[0].legend(handles=[Patch(color=PCOL[k], label=k) for k in PCOL], fontsize=8, loc="upper right")
fig.tight_layout()
out = "/scratch/euh7ys/climate_xai/plots/enso_children.png"
fig.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
