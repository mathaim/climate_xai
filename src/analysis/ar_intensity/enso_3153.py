"""Ground concept 3153 (extreme E. Australia summer ARs) in ENSO: do its strong firings
cluster in La Nina summers? DJF ONI from NOAA CPC (oni.ascii.txt)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
ONI = {1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,
       1988:0.81,1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,
       1996:-0.90,1997:-0.50,1998:2.24,1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,
       2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,2009:-0.85,2010:1.50,2011:-1.31,
       2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
PCOL = {"La Nina":"#2b6cb0","Neutral":"#9e9e9e","El Nino":"#c0392b"}
def phase(o): return "La Nina" if o<=-0.5 else ("El Nino" if o>=0.5 else "Neutral")
def main():
    t = np.load(f"{TRACK}/track_matry_E_Australia.npz")
    a = t["A_max"][:,3153].astype(float); ivt = t["ivt"].astype(float); ti = t["tindex"]
    ok = np.isfinite(ivt); a = a[ok]; ti = ti[ok]
    d = AR_START + pd.to_timedelta(6*(ti-1), unit="h"); yr = d.year.values; mo = d.month.values
    syr = yr + (mo==12)                                 # Dec belongs to next austral summer
    fire = a >= np.quantile(a, 0.99)                    # top-1% = strong firings
    rows = []
    for y in sorted(set(syr.tolist())):
        if y not in ONI: continue
        m = (syr==y) & np.isin(mo,[12,1,2,3])
        rows.append((y, int((fire & m).sum()), ONI[y], phase(ONI[y])))
    yrs = np.array([r[0] for r in rows]); cnt = np.array([r[1] for r in rows])
    oni = np.array([r[2] for r in rows]); ph = [r[3] for r in rows]
    ln = cnt[[p=="La Nina" for p in ph]]; en = cnt[[p=="El Nino" for p in ph]]
    r = float(np.corrcoef(cnt, oni)[0,1])
    print(f"mean strong 3153 summer firings:  La Nina={ln.mean():.1f}  El Nino={en.mean():.1f}  ratio={ln.mean()/max(en.mean(),1e-9):.1f}x")
    print(f"corr(firings, DJF ONI) = {r:.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(16,5), gridspec_kw={"width_ratios":[2.3,1]})
    ax[0].bar(yrs, cnt, color=[PCOL[p] for p in ph])
    ax[0].set_xlabel("Austral summer (year)"); ax[0].set_ylabel("Concept 3153 strong firings (Dec\u2013Mar)")
    ax[0].legend(handles=[Patch(color=PCOL[k],label=k) for k in PCOL], fontsize=9)
    ax[1].scatter(oni, cnt, c=[PCOL[p] for p in ph], s=40, edgecolor="k", lw=.4)
    ax[1].axvline(-0.5, ls="--", c="#2b6cb0", lw=.8); ax[1].axvline(0.5, ls="--", c="#c0392b", lw=.8)
    ax[1].set_xlabel("DJF ONI"); ax[1].set_ylabel("3153 strong firings"); ax[1].set_title(f"r = {r:.2f}")
    fig.tight_layout(); fig.savefig(f"{TRACK}/enso_3153.png", dpi=180, bbox_inches="tight")
    print("saved", f"{TRACK}/enso_3153.png")
if __name__ == "__main__":
    main()
