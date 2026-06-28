"""Cross-layer refinement: peak |r| for intensity (vs IVT) and ENSO (DJF vs ONI) per
region, per SAE/layer, plus the Matryoshka nested group of the top concept. Tests whether
concepts sharpen with depth and migrate toward the core."""
import numpy as np, pandas as pd
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; AR_START=pd.Timestamp("1979-01-01")
ONI={1980:0.59,1981:-0.26,1982:-0.05,1983:2.18,1984:-0.60,1985:-1.04,1986:-0.49,1987:1.23,1988:0.81,
1989:-1.69,1990:0.14,1991:0.41,1992:1.71,1993:0.09,1994:0.06,1995:0.96,1996:-0.90,1997:-0.50,1998:2.24,
1999:-1.55,2000:-1.66,2001:-0.68,2002:-0.15,2003:0.92,2004:0.37,2005:0.64,2006:-0.85,2007:0.66,2008:-1.64,
2009:-0.85,2010:1.50,2011:-1.31,2012:-0.72,2013:-0.29,2014:-0.28,2015:0.69,2016:2.63,2017:-0.19}
PREF={"plain_L0":"track_plain_L0","plain_L8":"track_pool","plain_L15":"track_plain_L15",
      "matry_L0":"track_matry_L0","matry_L8":"track_matry","matry_L15":"track_matry_L15"}
def grp(i): return "G0" if i<256 else "G1" if i<512 else "G2" if i<1024 else "G3" if i<2048 else "G4"
def pcol(A,y):
    a=A-A.mean(0,keepdims=True); yi=y-y.mean()
    den=np.sqrt((a**2).sum(0)*(yi**2).sum())
    return np.divide((a*yi[:,None]).sum(0),den,out=np.full(A.shape[1],np.nan),where=den>1e-9)
def main():
    print(f"{'SAE':10}{'region':13}{'int|r|':>8}{'int_c':>7}{'g':>4}{'enso|r|':>9}{'enso_c':>8}{'g':>4}")
    for sae,pref in PREF.items():
        for r in REGIONS:
            try: t=np.load(f"{TRACK}/{pref}_{r}.npz")
            except Exception: print(f"{sae:10}{r:13}  MISSING"); continue
            ivt=t["ivt"].astype(float); ti=t["tindex"]; ok=np.isfinite(ivt)
            A=np.asarray(t["A_max"][ok],dtype=np.float32); iv=ivt[ok].astype(np.float32); tio=ti[ok]; del t
            ri=np.abs(pcol(A,iv)); ic=int(np.nanargmax(ri)); ipk=float(ri[ic])
            d=AR_START+pd.to_timedelta(6*(tio-1),unit="h"); yr=d.year.values; mo=d.month.values
            djfy=yr+(mo==12); djf=np.isin(mo,[12,1,2]); years=[y for y in sorted(set(djfy[djf].tolist())) if y in ONI]
            M=np.zeros((len(years),A.shape[1]),np.float32)
            for j,y in enumerate(years):
                m=djf&(djfy==y)
                if m.any(): M[j]=A[m].mean(0)
            oni=np.array([ONI[y] for y in years],np.float32)
            re=np.abs(pcol(M,oni)); ec=int(np.nanargmax(re)); epk=float(re[ec]); mat="matry" in sae
            print(f"{sae:10}{r:13}{ipk:>8.2f}{ic:>7}{(grp(ic) if mat else '-'):>4}{epk:>9.2f}{ec:>8}{(grp(ec) if mat else '-'):>4}")
            del A,M
    print("DONE")
if __name__=="__main__": main()
