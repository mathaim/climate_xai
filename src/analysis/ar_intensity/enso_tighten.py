"""Tighten the children-ENSO result: mean DJF activation (cleaner) vs top-1% count, per child/region + dipole."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
ONI = {1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,
       1988:0.81,1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,
       1996:-0.90,1997:-0.50,1998:2.24,1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,
       2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,2009:-0.85,2010:1.50,2011:-1.31,
       2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
YEARS = sorted(ONI); oni = np.array([ONI[y] for y in YEARS])
def series(c, region, metric):
    t = np.load(f"{TRACK}/track_matry_{region}.npz")
    a = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float); ti = t["tindex"]
    ok = np.isfinite(ivt); a, ti = a[ok], ti[ok]
    d = AR_START + pd.to_timedelta(6*(ti-1), unit="h"); yr = d.year.values; mo = d.month.values
    syr = yr + (mo == 12); thr = np.quantile(a, 0.99); out = []
    for y in YEARS:
        sel = (syr == y) & np.isin(mo, [12,1,2,3])
        out.append(a[sel].mean() if (metric == "mean" and sel.sum()) else
                   (int((a[sel] >= thr).sum()) if metric == "count" else np.nan))
    return np.array(out, float)
def r(x): m = np.isfinite(x); return float(np.corrcoef(x[m], oni[m])[0, 1])
def z(x): return (x - np.nanmean(x)) / np.nanstd(x)
for metric in ["count", "mean"]:
    NHa, NHe, SH = series(3483,"W_N_America",metric), series(3483,"W_Europe",metric), series(3153,"E_Australia",metric)
    NH = (z(NHa) + z(NHe)) / 2; D = NH - z(SH)
    print(f"=== metric = {metric} ===")
    print(f"  3153/E.Aus   r = {r(SH):+.2f}")
    print(f"  3483/W.N.Am  r = {r(NHa):+.2f}")
    print(f"  3483/W.Eu    r = {r(NHe):+.2f}")
    print(f"  NH combined  r = {r(NH):+.2f}")
    print(f"  dipole       r = {r(D):+.2f}")
# figure: mean-activation, NH child vs SH child
NHa, NHe, SH = [series(c, rg, "mean") for c, rg in [(3483,"W_N_America"),(3483,"W_Europe"),(3153,"E_Australia")]]
NH = (z(NHa) + z(NHe)) / 2; SHz = z(SH)
col = ["#2b6cb0" if o<=-0.5 else "#c0392b" if o>=0.5 else "#9e9e9e" for o in oni]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a_, yv, ttl in [(ax[0], NH, f"NH child 3483 (W.N.Am + W.Eu)\nr = {r(NH):+.2f}"),
                    (ax[1], SHz, f"SH child 3153 (E. Australia)\nr = {r(SH):+.2f}")]:
    a_.scatter(oni, yv, c=col, s=55, edgecolor="k", lw=.4)
    b, a0 = np.polyfit(oni, yv, 1); xs = np.linspace(oni.min(), oni.max(), 10); a_.plot(xs, a0+b*xs, "k-", lw=1.2)
    a_.axhline(0, color="#999", lw=.6); a_.axvline(-0.5, ls="--", c="#2b6cb0", lw=.6); a_.axvline(0.5, ls="--", c="#c0392b", lw=.6)
    a_.set_xlabel("DJF Nino 3.4 (ONI)"); a_.set_title(ttl, fontsize=11)
ax[0].set_ylabel("standardized mean DJF activation")
ax[0].legend(handles=[Patch(color="#2b6cb0",label="La Nina"),Patch(color="#9e9e9e",label="Neutral"),Patch(color="#c0392b",label="El Nino")], fontsize=8)
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/enso_children_mean.png", dpi=180, bbox_inches="tight"); print("saved")
