"""ENSO organization: how many ENSO-aligned latents per region, and how mutually redundant
they are (mean pairwise |activation correlation|), for a given SAE track. High redundancy =
ENSO split/duplicated across latents; low = organized into fewer distinct directions."""
import sys, numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; AR_START=pd.Timestamp("1979-01-01")
ONI={1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,1988:0.81,
1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,1996:-0.90,1997:-0.50,1998:2.24,
1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,
2009:-0.85,2010:1.50,2011:-1.31,2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
PREFIX=sys.argv[1]; K=8
def main():
    print(f"=== {PREFIX} ===")
    for r in REGIONS:
        t=np.load(f"{TRACK}/{PREFIX}_{r}.npz"); A=t["A_max"]; ti=t["tindex"]
        d=AR_START+pd.to_timedelta(6*(ti-1),unit="h"); yr=d.year.values; mo=d.month.values
        djfy=yr+(mo==12); djf=np.isin(mo,[12,1,2])
        years=[y for y in sorted(set(djfy[djf].tolist())) if y in ONI]; oni=np.array([ONI[y] for y in years])
        M=np.zeros((len(years),4096))
        for i,y in enumerate(years):
            m=djf&(djfy==y); 
            if m.any(): M[i]=A[m].mean(0)
        del t,A
        Mc=M-M.mean(0); oc=oni-oni.mean()
        den=np.sqrt((Mc**2).sum(0)*(oc**2).sum()); rr=np.divide((Mc*oc[:,None]).sum(0),den,out=np.zeros(4096),where=den>1e-9)
        nstrong=int((np.abs(rr)>0.6).sum()); top=np.argsort(np.abs(rr))[::-1][:K]
        sub=M[:,top]; cc=np.corrcoef(sub.T); off=cc[~np.eye(K,dtype=bool)]
        print(f"  {r:14} peak|r|={np.abs(rr).max():.2f}  #(|r|>0.6)={nstrong:3d}  redundancy(mean|corr| top{K})={np.nanmean(np.abs(off)):.2f}")
    print()
if __name__=="__main__": main()
