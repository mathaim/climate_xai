"""Validate the top ENSO-correlated concepts: DJF mean activation vs DJF ONI per year,
with leave-one-out correlation range to check the relationship is not outlier-driven."""
import numpy as np, pandas as pd
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
ONI = {1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,
       1988:0.81,1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,
       1996:-0.90,1997:-0.50,1998:2.24,1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,
       2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,2009:-0.85,2010:1.50,2011:-1.31,
       2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
CANDS = [("E_Australia",1314,"El Nino"),("E_Australia",2532,"La Nina"),
         ("W_N_America",186,"El Nino"),("W_N_America",1171,"La Nina"),
         ("W_S_America",3919,"El Nino"),("W_S_America",860,"La Nina")]
def main():
    byreg = defaultdict(list)
    for r,c,lab in CANDS: byreg[r].append((c,lab))
    res = []
    for r, clist in byreg.items():
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); A = t["A_max"]; ti = t["tindex"]
        d = AR_START + pd.to_timedelta(6*(ti-1), unit="h"); yr = d.year.values; mo = d.month.values
        djfy = yr + (mo==12); djf = np.isin(mo,[12,1,2])
        years = [y for y in sorted(set(djfy[djf].tolist())) if y in ONI]
        oni = np.array([ONI[y] for y in years])
        for c, lab in clist:
            ac = A[:,c]; series = np.array([ac[djf & (djfy==y)].mean() for y in years])
            rr = float(np.corrcoef(series, oni)[0,1])
            loo = [float(np.corrcoef(np.delete(series,i), np.delete(oni,i))[0,1]) for i in range(len(years))]
            res.append((r,c,lab,oni,series,rr,min(loo),max(loo)))
        del t, A
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax,(r,c,lab,oni,series,rr,lo,hi) in zip(axes.flat, res):
        col = ["#c0392b" if o>=0.5 else "#2b6cb0" if o<=-0.5 else "#9e9e9e" for o in oni]
        ax.scatter(oni, series, c=col, s=45, edgecolor="k", lw=.4)
        ax.axvline(-0.5, ls="--", c="#2b6cb0", lw=.6); ax.axvline(0.5, ls="--", c="#c0392b", lw=.6)
        ax.set_title(f"{r}  c{c} ({lab})\nr={rr:.2f}   leave-one-out [{lo:.2f}, {hi:.2f}]", fontsize=11)
        ax.set_xlabel("DJF ONI"); ax.set_ylabel("DJF mean activation")
        print(f"{r} c{c} {lab}: r={rr:.2f} LOO[{lo:.2f},{hi:.2f}]")
    fig.tight_layout(); fig.savefig(f"{TRACK}/enso_concepts_validate.png", dpi=170, bbox_inches="tight")
    print("saved", f"{TRACK}/enso_concepts_validate.png")
if __name__ == "__main__":
    main()
