"""Test C: concept generality by nested group, Matryoshka L0/L8/L15 (+ Plain L8 control)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.sae_features import load_sae
from src.analysis.ar_intensity.regions import REGIONS
RES="results/ar_intensity"; OUT=f"{RES}/baseline"; PLOTS=f"{RES}/plots"; DEF="magnitude"
def nreg_of(SAE):
    no=np.load(f"{OUT}/nonar_rates_{SAE}.npz")
    F,md=load(SAE,f"region_{DEF}"); reg=md.region.to_numpy()
    magar={r:F[reg==r].mean(0) for r in REGIONS}
    ratio={r:magar[r]/np.maximum(no[f"{r}_{DEF}_no"],1e-3) for r in REGIONS}
    act={r:(ratio[r]>1.1)&(magar[r]>np.median(magar[r][magar[r]>0])) for r in REGIONS}
    return np.vstack([act[r] for r in REGIONS]).sum(0)
def main():
    pre=[int(x) for x in load_sae("matry_L0","cpu")[0].group_sizes]; b=[0]+pre
    groups=[(b[i],b[i+1]) for i in range(len(pre))]
    todo=["matry_L0","matry_L8","matry_L15","plain_L8"]; rows=[]; glob={}
    for SAE in todo:
        n=nreg_of(SAE); glob[SAE]=int((n>=3).sum())
        for gi,(lo,hi) in enumerate(groups):
            seg=n[lo:hi]; na=int((seg>=1).sum())
            rows.append(dict(sae=SAE,group=gi,n_active=na,
                             mean_nregions=float(seg[seg>=1].mean()) if na else 0.0,
                             pct_global=100*float((seg>=3).sum())/max(na,1)))
    df=pd.DataFrame(rows); df.to_csv(f"{RES}/generality_by_group.csv",index=False)
    print("global (>=3 region) parent-concept counts:",glob)
    print(df.pivot_table(index="sae",columns="group",values="mean_nregions").round(2))
    plt.rcParams.update({"font.size":12,"axes.labelsize":13,"legend.fontsize":10})
    fig,ax=plt.subplots(figsize=(7.8,5.2))
    COL={"matry_L0":"#9ecae1","matry_L8":"#4292c6","matry_L15":"#08519c"}
    for SAE in ["matry_L0","matry_L8","matry_L15"]:
        d=df[df.sae==SAE].sort_values("group")
        ax.plot(d.group,d.mean_nregions,"o-",color=COL[SAE],lw=2.4,ms=7,label=f"Matryoshka SAE L{SAE.split('_L')[1]}")
    d=df[df.sae=="plain_L8"].sort_values("group")
    ax.plot(d.group,d.mean_nregions,"s--",color="#c0392b",lw=2,ms=6,label="Plain SAE L8 (index control)")
    ax.set_xticks(range(5)); ax.set_xticklabels(["G0\n(core)","G1","G2","G3","G4\n(outer)"])
    ax.set_xlabel("Nested Group (Core to Outer)"); ax.set_ylabel("Mean Generality (regions per concept)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(f"{PLOTS}/generality_by_group.png",dpi=200,bbox_inches="tight")
    print("SAVED generality_by_group.png + generality_by_group.csv")
if __name__=="__main__": main()
