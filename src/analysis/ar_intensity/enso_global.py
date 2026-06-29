"""Region-free test: each child's TOTAL activation across all 4 AR regions per DJF season vs Nino 3.4."""
import numpy as np, pandas as pd
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
ONI = {1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,
       1988:0.81,1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,
       1996:-0.90,1997:-0.50,1998:2.24,1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,
       2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,2009:-0.85,2010:1.50,2011:-1.31,
       2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
YEARS = sorted(ONI); oni = np.array([ONI[y] for y in YEARS])
REGIONS = ["W_N_America", "W_Europe", "W_S_America", "E_Australia"]
def season_mean(c, region):
    t = np.load(f"{TRACK}/track_matry_{region}.npz")
    a = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float); ti = t["tindex"]
    ok = np.isfinite(ivt); a, ti = a[ok], ti[ok]
    d = AR_START + pd.to_timedelta(6*(ti-1), unit="h"); yr = d.year.values; mo = d.month.values
    syr = yr + (mo == 12)
    return {y: a[(syr == y) & np.isin(mo, [12,1,2,3])].mean() for y in YEARS
            if ((syr == y) & np.isin(mo, [12,1,2,3])).sum()}
def total(c):
    tot = np.zeros(len(YEARS))
    for region in REGIONS:
        sm = season_mean(c, region)
        tot += np.array([sm.get(y, 0.0) for y in YEARS])
    return tot
for c, name in [(3153, "3153 (E.Aus child)"), (3483, "3483 (NH child)")]:
    s = total(c); r = float(np.corrcoef(s, oni)[0, 1])
    print(f"{name}:  r(total 4-region activation, Nino3.4) = {r:+.2f}")
