"""ENSO-phase composite: does each child fire more in El Nino vs La Nina winters? (mean DJF activation + test)"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
ONI = {1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,
       1988:0.81,1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,
       1996:-0.90,1997:-0.50,1998:2.24,1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,
       2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,2009:-0.85,2010:1.50,2011:-1.31,
       2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
YEARS = sorted(ONI)
PH = ["El Nino", "Neutral", "La Nina"]; PCOL = {"El Nino":"#c0392b","Neutral":"#9e9e9e","La Nina":"#2b6cb0"}
def phase(o): return "El Nino" if o >= 0.5 else ("La Nina" if o <= -0.5 else "Neutral")
try:
    from scipy import stats
    def pval(a, b): return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue, "Mann-Whitney"
except Exception:
    from math import erf, sqrt
    def pval(a, b):
        a, b = np.array(a), np.array(b)
        t = (a.mean()-b.mean())/sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
        return 2*(1-0.5*(1+erf(abs(t)/sqrt(2)))), "Welch t (approx)"
def season_mean(c, region):
    t = np.load(f"{TRACK}/track_matry_{region}.npz")
    a = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float); ti = t["tindex"]
    ok = np.isfinite(ivt); a, ti = a[ok], ti[ok]
    d = AR_START + pd.to_timedelta(6*(ti-1), unit="h"); yr = d.year.values; mo = d.month.values
    syr = yr + (mo == 12)
    out = {}
    for y in YEARS:
        m = (syr == y) & np.isin(mo, [12,1,2,3])
        if m.sum(): out[y] = a[m].mean()
    return out
PAIRS = [(3483, "W_N_America", "NH child 3483  \u00b7  W. North America"),
         (3153, "E_Australia", "SH child 3153  \u00b7  E. Australia")]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (c, region, title) in zip(axes, PAIRS):
    sm = season_mean(c, region)
    byph = {p: [sm[y] for y in YEARS if y in sm and phase(ONI[y]) == p] for p in PH}
    means = [np.mean(byph[p]) for p in PH]; sems = [np.std(byph[p])/np.sqrt(len(byph[p])) for p in PH]
    ax.bar(PH, means, yerr=sems, color=[PCOL[p] for p in PH], capsize=5, edgecolor="k", lw=.5)
    p, tname = pval(byph["El Nino"], byph["La Nina"]); ratio = np.mean(byph["El Nino"]) / np.mean(byph["La Nina"])
    ax.set_title(f"{title}\nEl Nino / La Nina = {ratio:.2f}x   ({tname} p = {p:.3f})", fontsize=10)
    ax.set_ylabel("mean DJF concept activation")
    print(f"{title}: ElNino={np.mean(byph['El Nino']):.3f}  LaNina={np.mean(byph['La Nina']):.3f}  ratio={ratio:.2f}x  p={p:.3f}  (n El/Neu/La = {len(byph['El Nino'])}/{len(byph['Neutral'])}/{len(byph['La Nina'])})")
fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/enso_composite.png", dpi=180, bbox_inches="tight"); print("saved")
