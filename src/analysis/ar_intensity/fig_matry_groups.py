"""Matryoshka nested-group intensity concentration, publication styling (from matry_group.csv)."""
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES="results/ar_intensity"; PLOTS=f"{RES}/plots"; LAYERS=[0,8,15]
VARIANTS=[f"{s}_{d}" for s in ["region","global"] for d in ["binary","magnitude","top10"]]
TITLE={"region_binary":"Region: Binary","region_magnitude":"Region: Magnitude","region_top10":"Region: Top 10%",
       "global_binary":"Global: Binary","global_magnitude":"Global: Magnitude","global_top10":"Global: Top 10%"}
COL={0:"#2b6cb0",8:"#d8722c",15:"#1a7a3a"}
plt.rcParams.update({"font.size":12,"axes.titlesize":13,"axes.labelsize":13,"legend.fontsize":10})
def panels(variants, fname, ncol):
    agg=pd.read_csv(f"{RES}/matry_group.csv").groupby(["variant","layer","group"]).mean_abs_r.mean().reset_index()
    nrow=(len(variants)+ncol-1)//ncol
    fig,ax=plt.subplots(nrow,ncol,figsize=(4.5*ncol,3.6*nrow),sharex=True,squeeze=False)
    for i,v in enumerate(variants):
        a=ax.flat[i]; dv=agg[agg.variant==v]
        for L in LAYERS:
            d=dv[dv.layer==L].sort_values("group")
            a.plot(d.group,d.mean_abs_r,"o-",color=COL[L],lw=2,ms=5,label=f"Layer {L}")
        a.set_title(TITLE[v]); a.set_xticks(range(5)); a.set_xticklabels(["G0","G1","G2","G3","G4"]); a.grid(alpha=.3)
    ax.flat[0].legend(title="Depth")
    fig.supylabel("Mean |r| with Intensity"); fig.supxlabel("Nested Group (Core $\\rightarrow$ Outer)")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/{fname}",dpi=200,bbox_inches="tight"); plt.close(fig)
    print("SAVED",fname)
def main():
    panels(VARIANTS,"matry_groups_by_variant.png",3)
    panels([f"region_{d}" for d in ["binary","magnitude","top10"]],"matry_groups_region.png",3)
if __name__=="__main__": main()
