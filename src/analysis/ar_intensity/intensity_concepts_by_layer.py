"""Count of low- vs high-intensity-tuned concepts by layer (from concept_tuning.csv, no re-encode)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES="results/ar_intensity"; PLOTS=f"{RES}/plots"; LAYERS=[0,8,15]; VAR="region_magnitude"
def main():
    t=pd.read_csv(f"{RES}/concept_tuning.csv"); t=t[t.variant==VAR].copy()
    c=t.groupby(["arch","layer","region","preferred_bin"]).concept.count().reset_index(name="n")
    m=c.groupby(["arch","layer","preferred_bin"]).n.mean().reset_index()   # mean over the 4 regions
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for a,arch in zip(ax,["plain","matry"]):
        for b,lab,col in [(0,"low-intensity (bottom 10%)","#2b6cb0"),(3,"high-intensity (top 10%)","#c0392b")]:
            d=m[(m.arch==arch)&(m.preferred_bin==b)].set_index("layer").reindex(LAYERS).fillna(0)
            a.plot(LAYERS,d.n,"o-",color=col,lw=2,label=lab)
        a.set_title("vanilla" if arch=="plain" else "matryoshka"); a.set_xticks(LAYERS)
        a.set_xlabel("processor layer"); a.grid(alpha=.3)
    ax[0].set_ylabel("# concepts (mean over regions)"); ax[0].legend()
    fig.suptitle("Low- vs high-intensity-tuned concepts by layer (preferred intensity bin)")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/intensity_concepts_by_layer.png",dpi=160,bbox_inches="tight")
    print("=== mean # concepts per region by preferred bin (0=bottom10 ... 3=top10) ===")
    print(m.pivot_table(index=["arch","layer"],columns="preferred_bin",values="n").round(1))
    print("SAVED intensity_concepts_by_layer.png")
if __name__=="__main__": main()
