"""Low- vs high-intensity concept counts by layer: region-averaged and per-region (from concept_tuning.csv)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
RES="results/ar_intensity"; PLOTS=f"{RES}/plots"; LAYERS=[0,8,15]; VAR="region_magnitude"
REGIONS=["W_N_America","W_Europe","W_S_America","E_Australia"]
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe",
      "W_S_America":"Western South America","E_Australia":"Eastern Australia"}
COLORS={"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
ARCH={"plain":"Plain SAE","matry":"Matryoshka SAE"}
plt.rcParams.update({"font.size":12,"axes.titlesize":14,"axes.labelsize":13,"legend.fontsize":10})
def main():
    t=pd.read_csv(f"{RES}/concept_tuning.csv"); t=t[t.variant==VAR].copy()
    c=t.groupby(["arch","layer","region","preferred_bin"]).concept.count().reset_index(name="n")
    # ---- region-averaged ----
    m=c.groupby(["arch","layer","preferred_bin"]).n.mean().reset_index()
    fig,ax=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for a,arch in zip(ax,["plain","matry"]):
        for b,lab,col in [(3,"High-Intensity (Top 10%)","#c0392b"),(0,"Low-Intensity (Bottom 10%)","#2b6cb0")]:
            d=m[(m.arch==arch)&(m.preferred_bin==b)].set_index("layer").reindex(LAYERS).fillna(0)
            a.plot(LAYERS,d.n,"o-",color=col,lw=2.2,ms=6,label=lab)
        a.set_title(ARCH[arch]); a.set_xticks(LAYERS); a.set_xlabel("Processor Layer"); a.grid(alpha=.3)
    ax[0].set_ylabel("Number of Concepts (Mean over Regions)"); ax[0].legend(frameon=True)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/intensity_concepts_by_layer.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    # ---- per-region (high solid, low dotted, color per region) ----
    fig,ax=plt.subplots(1,2,figsize=(12.5,5),sharey=True)
    for a,arch in zip(ax,["plain","matry"]):
        for r in REGIONS:
            for b,ls in [(3,"-"),(0,":")]:
                d=c[(c.arch==arch)&(c.region==r)&(c.preferred_bin==b)].set_index("layer").reindex(LAYERS).fillna(0)
                a.plot(LAYERS,d.n,ls,color=COLORS[r],lw=2.2,marker="o",ms=5)
        a.set_title(ARCH[arch]); a.set_xticks(LAYERS); a.set_xlabel("Processor Layer"); a.grid(alpha=.3)
    ax[0].set_ylabel("Number of Concepts")
    reg_h=[Line2D([0],[0],color=COLORS[r],lw=2.4,label=NAME[r]) for r in REGIONS]
    sty_h=[Line2D([0],[0],color="#444",lw=2.4,ls="-",label="High-Intensity (Top 10%)"),
           Line2D([0],[0],color="#444",lw=2.4,ls=":",label="Low-Intensity (Bottom 10%)")]
    ax[0].legend(handles=reg_h,loc="upper left",title="Region",frameon=True)
    ax[1].legend(handles=sty_h,loc="upper left",frameon=True)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/intensity_concepts_by_layer_regional.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    print("SAVED intensity_concepts_by_layer.png + intensity_concepts_by_layer_regional.png")
if __name__=="__main__": main()
