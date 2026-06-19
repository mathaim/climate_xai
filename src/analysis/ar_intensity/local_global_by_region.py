"""Local (solid) vs global (dotted) concepts per region across layers (Plain SAE baseline)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.regions import REGIONS
RES="results/ar_intensity"; OUT=f"{RES}/baseline"; PLOTS=f"{RES}/plots"; LAYERS=[0,8,15]; DEF="magnitude"
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe",
      "W_S_America":"Western South America","E_Australia":"Eastern Australia"}
COLORS={"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
plt.rcParams.update({"font.size":12,"axes.labelsize":13,"legend.fontsize":9})
def sets(layer):
    SAE=f"plain_L{layer}"; no=np.load(f"{OUT}/nonar_rates_{SAE}.npz")
    F,md=load(SAE,f"region_{DEF}"); reg=md.region.to_numpy()
    magar={r:F[reg==r].mean(0) for r in REGIONS}
    ratio={r:magar[r]/np.maximum(no[f"{r}_{DEF}_no"],1e-3) for r in REGIONS}
    act={r:(ratio[r]>1.1)&(magar[r]>np.median(magar[r][magar[r]>0])) for r in REGIONS}
    nreg=np.vstack([act[r] for r in REGIONS]).sum(0)
    return ({r:int((act[r]&(nreg==1)).sum()) for r in REGIONS},
            {r:int((act[r]&(nreg>=3)).sum()) for r in REGIONS})
def main():
    LOC={r:[] for r in REGIONS}; GLO={r:[] for r in REGIONS}
    for L in LAYERS:
        loc,glo=sets(L)
        for r in REGIONS: LOC[r].append(loc[r]); GLO[r].append(glo[r])
    fig,ax=plt.subplots(figsize=(7.5,5.2))
    for r in REGIONS:
        ax.plot(LAYERS,LOC[r],"-",color=COLORS[r],lw=2.3,marker="o",ms=6)
        ax.plot(LAYERS,GLO[r],":",color=COLORS[r],lw=2.3,marker="o",ms=6)
    ax.set_xticks(LAYERS); ax.set_xlabel("Processor Layer"); ax.set_ylabel("Number of Concepts"); ax.grid(alpha=.3)
    reg_h=[Line2D([0],[0],color=COLORS[r],lw=2.4,label=NAME[r]) for r in REGIONS]
    sty_h=[Line2D([0],[0],color="#444",lw=2.4,ls="-",label="Local (1 region)"),
           Line2D([0],[0],color="#444",lw=2.4,ls=":",label="Global ($\\geq$3 regions)")]
    l1=ax.legend(handles=reg_h,loc="upper left",title="Region"); ax.add_artist(l1)
    ax.legend(handles=sty_h,loc="upper center")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/local_global_by_region.png",dpi=200,bbox_inches="tight")
    print("Local:");  [print(f"  {r:13}",LOC[r]) for r in REGIONS]
    print("Global:"); [print(f"  {r:13}",GLO[r]) for r in REGIONS]
    print("(layers",LAYERS,") SAVED local_global_by_region.png")
if __name__=="__main__": main()
