"""Concepts per region across layers (one line per region, no intensity-bin split)."""
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES="results/ar_intensity"; PLOTS=f"{RES}/plots"; LAYERS=[0,8,15]; VAR="region_magnitude"
REGIONS=["W_N_America","W_Europe","W_S_America","E_Australia"]
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe",
      "W_S_America":"Western South America","E_Australia":"Eastern Australia"}
COLORS={"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
ARCH={"plain":"Plain SAE","matry":"Matryoshka SAE"}
plt.rcParams.update({"font.size":12,"axes.titlesize":14,"axes.labelsize":13,"legend.fontsize":10})
def main():
    s=pd.read_csv(f"{RES}/corr_summary.csv"); s=s[s.variant==VAR].copy(); s["n"]=s.frac_strong*4096
    fig,ax=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for a,arch in zip(ax,["plain","matry"]):
        for r in REGIONS:
            d=s[(s.arch==arch)&(s.region==r)].set_index("layer").reindex(LAYERS)
            a.plot(LAYERS,d.n,"o-",color=COLORS[r],lw=2.2,ms=6,label=NAME[r])
        a.set_title(ARCH[arch]); a.set_xticks(LAYERS); a.set_xlabel("Processor Layer"); a.grid(alpha=.3)
    ax[0].set_ylabel("Number of Concepts"); ax[0].legend(title="Region")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/concepts_by_region.png",dpi=200,bbox_inches="tight")
    print(s.pivot_table(index=["arch","region"],columns="layer",values="n").round(0))
    print("SAVED concepts_by_region.png")
if __name__=="__main__": main()
