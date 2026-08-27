"""Concept activation vs region max-IVT scatter over the full 1979-2017 record.
Two figures: (A) every 6h timestep, (B) one point per AR event peak. 2x2 region champions."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TR="/scratch/euh7ys/climate_xai/concept_ivt"; PLOTS="/scratch/euh7ys/climate_xai/plots"
REGIONS=["W_N_America","W_Europe","W_S_America","E_Australia"]
CONCEPT={"W_N_America":1592,"W_Europe":2948,"W_S_America":3218,"E_Australia":3720}
TITLE={"W_N_America":"Western North America","W_Europe":"Western Europe",
       "W_S_America":"Western South America","E_Australia":"Eastern Australia"}
AR_THR=250.0
def load(r):
    d=np.load(f"{TR}/track_pool_{r}.npz"); o=np.argsort(d["tindex"])
    return d["A_mean"][o], d["A_max"][o], d["ivt"][o], d["tindex"][o]
def event_peaks(ivt,ti):
    m=ivt>=AR_THR; pk=[]; i=0; n=len(ivt)
    while i<n:
        if m[i]:
            j=i
            while j+1<n and m[j+1] and ti[j+1]==ti[j]+1: j+=1
            pk.append(i+int(np.argmax(ivt[i:j+1]))); i=j+1
        else: i+=1
    return np.array(pk)
# ---- verification: which activation reproduces the org-map correlation? ----
print(f"{'region':13}{'concept':>8}{'ts A_mean':>10}{'ts A_max':>10}{'ev A_mean':>10}{'ev A_max':>10}")
for r in REGIONS:
    Am,Ax,iv,ti=load(r); cc=CONCEPT[r]; pk=event_peaks(iv,ti)
    f=lambda a,y:np.corrcoef(a,y)[0,1]
    print(f"{r:13}{cc:8d}{f(Am[:,cc],iv):10.3f}{f(Ax[:,cc],iv):10.3f}{f(Am[pk,cc],iv[pk]):10.3f}{f(Ax[pk,cc],iv[pk]):10.3f}")
# ---- figures (default activation = A_mean; flip ACT to 'max' if that matches better) ----
ACT="max"
for per_event,tag in [(False,"timestep"),(True,"event")]:
    fig,axes=plt.subplots(2,2,figsize=(11,9.2))
    for ax,r in zip(axes.ravel(),REGIONS):
        Am,Ax,iv,ti=load(r); cc=CONCEPT[r]; A=Am if ACT=="mean" else Ax
        if per_event:
            pk=event_peaks(iv,ti); x=A[pk,cc]; y=iv[pk]; sc=dict(s=16,alpha=.55)
        else:
            x=A[:,cc]; y=iv; sc=dict(s=4,alpha=.06)
        ax.scatter(x,y,color="#185FA5",edgecolor="none",**sc)
        rr=float(np.corrcoef(x,y)[0,1]); b,a0=np.polyfit(x,y,1)
        xs=np.array([x.min(),x.max()]); ax.plot(xs,b*xs+a0,color="#c0392b",lw=1.7)
        ax.axhline(AR_THR,color="0.55",lw=.8,ls="--")
        ax.text(.04,.95,f"concept {cc}\n$r={rr:.2f}$  (n={len(x)})",transform=ax.transAxes,
                va="top",fontsize=10.5,bbox=dict(fc="white",ec="0.7",alpha=.85,pad=3))
        ax.set_title(TITLE[r],fontsize=12); ax.set_xlabel(f"concept {cc} activation")
        ax.set_ylabel("region max IVT (kg m$^{-1}$ s$^{-1}$)"); ax.grid(alpha=.22)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/scatter_concept_ivt_{tag}.png",dpi=170,bbox_inches="tight")
    print("saved scatter_concept_ivt_%s.png"%tag)
