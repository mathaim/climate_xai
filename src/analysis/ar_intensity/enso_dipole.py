"""Joint test: NH child (3483) minus SH child (3153) firing should track Nino 3.4 (the teleconnection dipole)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
ONI = {1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,
       1988:0.81,1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,
       1996:-0.90,1997:-0.50,1998:2.24,1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,
       2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,2009:-0.85,2010:1.50,2011:-1.31,
       2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
YEARS = sorted(ONI)
def djf_counts(c, region):
    t = np.load(f"{TRACK}/track_matry_{region}.npz")
    a = t["A_max"][:, c].astype(float); ivt = t["ivt"].astype(float); ti = t["tindex"]
    ok = np.isfinite(ivt); a, ti = a[ok], ti[ok]
    d = AR_START + pd.to_timedelta(6*(ti-1), unit="h"); yr = d.year.values; mo = d.month.values
    syr = yr + (mo == 12); fire = a >= np.quantile(a, 0.99)
    return {y: int((fire & ((syr == y) & np.isin(mo, [12,1,2,3]))).sum()) for y in YEARS}
def z(dct):
    v = np.array([dct.get(y, 0) for y in YEARS], float); return (v - v.mean()) / v.std()
zNH = (z(djf_counts(3483, "W_N_America")) + z(djf_counts(3483, "W_Europe"))) / 2
zSH = z(djf_counts(3153, "E_Australia"))
D = zNH - zSH; oni = np.array([ONI[y] for y in YEARS])
def r(x): return float(np.corrcoef(x, oni)[0, 1])
print(f"r(NH child, Nino3.4)   = {r(zNH):+.2f}")
print(f"r(SH child, Nino3.4)   = {r(zSH):+.2f}")
print(f"r(NH - SH dipole, N3.4) = {r(D):+.2f}   <-- joint test")
col = ["#2b6cb0" if o<=-0.5 else "#c0392b" if o>=0.5 else "#9e9e9e" for o in oni]
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(oni, D, c=col, s=55, edgecolor="k", lw=.4)
b, a0 = np.polyfit(oni, D, 1); xs = np.linspace(oni.min(), oni.max(), 10); ax.plot(xs, a0+b*xs, "k-", lw=1.3)
ax.axhline(0, color="#999", lw=.7); ax.axvline(-0.5, ls="--", c="#2b6cb0", lw=.7); ax.axvline(0.5, ls="--", c="#c0392b", lw=.7)
ax.set_xlabel("DJF Nino 3.4 (ONI)"); ax.set_ylabel("NH child (3483) minus SH child (3153)\nstandardized DJF firing")
ax.set_title(f"Teleconnection dipole of concept 99's children vs ENSO   (r = {r(D):+.2f})", fontsize=11)
fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/enso_dipole.png", dpi=180, bbox_inches="tight")
print("saved")
