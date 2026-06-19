"""Low- vs high-intensity concept counts by layer: region-averaged and per-region (from concept_tuning.csv)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
RES="results/ar_intensity"; PLOTS=f"{RES}/plots"; LAYERS=[0,8,15]; VAR="region_magnitude"
REGIONS=["W_N_America","W_Europe","W_S_America","E_Australia"]
COLORS={"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
def main():
    t=pd.read_csv(f"{RES}/concept_tuning.csv"); t=t[t.variant==VAR].copy()
    c=t.groupby(["arch","layer","region","preferred_bin"]).concept.count().reset_index(name="n")
    # ---- region-averaged (original) ----
    m=c.groupby(["arch","layer","preferred_bin"]).n.mean().reset_index()
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for a,arch in zip(ax,["plain","matry"]):
        for b,lab,col in [(0,"low-intensity (bottom 10%)","#2b6cb0"),(3,"high-intensity (top 10%)","#c0392b")]:
            d=m[(m.arch==arch)&(m.preferred_bin==b)].set_index("layer").reindex(LAYERS).fillna(0)
            a.plot(LAYERS,d.n,"o-",color=col,lw=2,label=lab)
        a.set_title("vanilla" if arch=="plain" else "matryoshka"); a.set_xticks(LAYERS); a.set_xlabel("processor layer"); a.grid(alpha=.3)
    ax[0].set_ylabel("# concepts (mean over regions)"); ax[0].legend()
    fig.suptitle("Low- vs high-intensity-tuned concepts by layer"); fig.tight_layout()
    fig.savefig(f"{PLOTS}/intensity_concepts_by_layer.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    # ---- per-region (high solid, low dotted, color per region) ----
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for a,arch in zip(ax,["plain","matry"]):
        for r in REGIONS:
            for b,ls in [(3,"-"),(0,":")]:
                d=c[(c.arch==arch)&(c.region==r)&(c.preferred_bin==b)].set_index("layer").reindex(LAYERS).fillna(0)
                a.plot(LAYERS,d.n,ls,color=COLORS[r],lw=2,marker="o",ms=4)
        a.set_title("vanilla" if arch=="plain" else "matryoshka"); a.set_xticks(LAYERS); a.set_xlabel("processor layer"); a.grid(alpha=.3)
    ax[0].set_ylabel("# concepts")
    reg_h=[Line2D([0],[0],color=COLORS[r],lw=2,label=r.replace("_"," ")) for r in REGIONS]
    sty_h=[Line2D([0],[0],color="#444",lw=2,ls="-",label="high-intensity (top 10%)"),
           Line2D([0],[0],color="#444",lw=2,ls=":",label="low-intensity (bottom 10%)")]
    ax[0].legend(handles=reg_h,loc="upper left",fontsize=8,title="region")
    ax[1].legend(handles=sty_h,loc="upper left",fontsize=8)
    fig.suptitle("Low- (dotted) vs high-intensity (solid) concepts by layer and region"); fig.tight_layout()
    fig.savefig(f"{PLOTS}/intensity_concepts_by_layer_regional.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("=== per-region counts (preferred_bin 0=bottom10, 3=top10) ===")
    print(c[c.preferred_bin.isin([0,3])].pivot_table(index=["arch","region","preferred_bin"],columns="layer",values="n").round(0))
    print("SAVED intensity_concepts_by_layer.png + intensity_concepts_by_layer_regional.png")
if __name__=="__main__": main()
