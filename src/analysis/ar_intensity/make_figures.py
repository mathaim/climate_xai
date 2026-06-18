"""Stage-5 figures from results/ar_intensity CSVs + corr arrays -> plots/."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES="results/ar_intensity"; PLOTS=f"{RES}/plots"; LAYERS=[0,8,15]
VARIANTS=[f"{s}_{d}" for s in ["region","global"] for d in ["binary","magnitude","top10"]]
REGIONS=["W_N_America","W_Europe","W_S_America","E_Australia"]; BINS=["bottom10","low_mid40","up_mid40","top10"]

def fig_decode():
    p=pd.read_csv(f"{RES}/decodability.csv").query("region=='pooled'")
    fig,ax=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for i,v in enumerate(VARIANTS):
        a=ax.flat[i]; d0=p[p.variant==v]
        for arch in ["plain","matry"]:
            d=d0[d0.arch==arch].sort_values("layer"); a.plot(d.layer,d.balanced_acc,"o-",label=arch)
        a.axhline(0.25,ls="--",c="gray"); a.set_title(v); a.set_xticks(LAYERS); a.grid(alpha=.3)
    ax.flat[0].legend(); fig.supylabel("balanced accuracy (pooled)"); fig.supxlabel("layer")
    fig.suptitle("AR-intensity decodability by firing variant (chance=0.25)")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/decodability_by_variant.png",dpi=200); plt.close(fig)

def fig_decode_region(variant="region_binary"):
    d0=pd.read_csv(f"{RES}/decodability.csv").query("variant==@variant and region!='pooled'")
    fig,ax=plt.subplots(1,4,figsize=(16,4),sharey=True)
    for a,r in zip(ax,REGIONS):
        dr=d0[d0.region==r]
        for arch in ["plain","matry"]:
            d=dr[dr.arch==arch].sort_values("layer"); a.plot(d.layer,d.balanced_acc,"o-",label=arch)
        a.axhline(0.25,ls="--",c="gray"); a.set_title(r); a.set_xticks(LAYERS); a.grid(alpha=.3)
    ax[0].legend(); ax[0].set_ylabel("balanced accuracy")
    fig.suptitle(f"Decodability by region ({variant})"); fig.tight_layout()
    fig.savefig(f"{PLOTS}/decodability_by_region.png",dpi=200); plt.close(fig)

def fig_matry_groups():
    agg=pd.read_csv(f"{RES}/matry_group.csv").groupby(["variant","layer","group"]).mean_abs_r.mean().reset_index()
    fig,ax=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for i,v in enumerate(VARIANTS):
        a=ax.flat[i]; dv=agg[agg.variant==v]
        for L in LAYERS:
            d=dv[dv.layer==L].sort_values("group"); a.plot(d.group,d.mean_abs_r,"o-",label=f"L{L}")
        a.set_title(v); a.set_xticks(range(5)); a.set_xticklabels(["G0","G1","G2","G3","G4"]); a.grid(alpha=.3)
    ax.flat[0].legend(title="depth"); fig.supylabel("mean |r| with intensity"); fig.supxlabel("nested group (core→outer)")
    fig.suptitle("Matryoshka: intensity across nested dictionary, by firing variant")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/matry_groups_by_variant.png",dpi=200); plt.close(fig)

def fig_corr_dist():
    fig,ax=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for i,v in enumerate(VARIANTS):
        a=ax.flat[i]; data=[]; labs=[]; pos=[]; x=1
        for L in LAYERS:
            for arch in ["plain","matry"]:
                name=f"{arch}_L{L}"; vals=[]
                for r in REGIONS:
                    f=f"{RES}/corr/{name}_{v}_{r}.npy"
                    if os.path.exists(f): vals.append(np.abs(np.load(f)))
                data.append(np.concatenate(vals) if vals else np.array([0.0])); labs.append(f"L{L}{arch[0]}"); pos.append(x); x+=1
            x+=0.5
        a.boxplot(data,positions=pos,showfliers=False,widths=.6)
        a.set_xticks(pos); a.set_xticklabels(labs,fontsize=7); a.set_title(v); a.grid(alpha=.3,axis="y")
    ax.flat[0].set_ylabel("|r| per concept"); fig.suptitle("Per-concept |r| with intensity (p=plain, m=matry)")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/corr_distribution.png",dpi=200); plt.close(fig)

def fig_preferred_bin(variant="region_binary"):
    t=pd.read_csv(f"{RES}/concept_tuning.csv").query("variant==@variant")
    g=t.groupby(["arch","layer","preferred_bin"]).concept.count().reset_index(name="n")
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for a,arch in zip(ax,["plain","matry"]):
        da=g[g.arch==arch]
        for L in LAYERS:
            d=da[da.layer==L].set_index("preferred_bin").reindex(range(len(BINS))).fillna(0)
            a.plot(range(len(BINS)),d.n,"o-",label=f"L{L}")
        a.set_xticks(range(len(BINS))); a.set_xticklabels(BINS,rotation=20,fontsize=8)
        a.set_title(arch); a.grid(alpha=.3); a.set_xlabel("preferred bin")
    ax[0].legend(title="depth"); ax[0].set_ylabel("# concepts (summed over regions)")
    fig.suptitle(f"Concept specialization by preferred intensity bin ({variant})")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/preferred_bin_{variant}.png",dpi=200); plt.close(fig)

def fig_extreme(sae="matry_L15", variant="region_top10", region="E_Australia", topn=12):
    d=pd.read_csv(f"{RES}/concept_tuning.csv").query("sae==@sae and variant==@variant and region==@region").copy()
    d["peak"]=d[["b0","b1","b2","b3"]].max(1); d=d[d.peak>d.peak.median()]
    top=d.sort_values("extreme_enrichment",ascending=False).head(topn)
    fig,a=plt.subplots(figsize=(7,4.5))
    for _,row in top.iterrows():
        prof=np.array([row.b0,row.b1,row.b2,row.b3]); s=prof.sum() or 1; prof=prof/s
        a.plot(range(4),prof,"o-",alpha=.7,label=f"c{int(row.concept)} ({row.extreme_enrichment:.1f}x)")
    a.set_xticks(range(4)); a.set_xticklabels(BINS,rotation=20); a.set_ylabel("normalized profile")
    a.legend(fontsize=7,ncol=2); a.set_title(f"Top extreme-AR concepts: {sae}/{variant}/{region}")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/extreme_concepts_{sae}_{region}.png",dpi=200); plt.close(fig)

def main():
    os.makedirs(PLOTS,exist_ok=True)
    fig_decode(); fig_decode_region(); fig_matry_groups(); fig_corr_dist()
    fig_preferred_bin("region_binary"); fig_preferred_bin("region_top10"); fig_extreme()
    print("FIGURES SAVED to",PLOTS)
if __name__=="__main__": main()
