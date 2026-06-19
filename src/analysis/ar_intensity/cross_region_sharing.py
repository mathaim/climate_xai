"""Cross-region concept-sharing (baseline L8 Plain SAE, magnitude active sets)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.regions import REGIONS
OUT="results/ar_intensity/baseline"; PLOTS="results/ar_intensity/plots"; SAE="plain_L8"; DEF="magnitude"
NAME={"W_N_America":"W. N. America","W_Europe":"W. Europe","W_S_America":"W. S. America","E_Australia":"E. Australia"}
def main():
    no=np.load(f"{OUT}/nonar_rates_{SAE}.npz")
    F,md=load(SAE,f"region_{DEF}"); reg=md.region.to_numpy()
    magar={r:F[reg==r].mean(0) for r in REGIONS}
    ratio={r:magar[r]/np.maximum(no[f"{r}_{DEF}_no"],1e-3) for r in REGIONS}
    active={r:(ratio[r]>1.1)&(magar[r]>np.median(magar[r][magar[r]>0])) for r in REGIONS}
    RL=list(REGIONS); n=len(RL); sizes={r:int(active[r].sum()) for r in RL}
    I=np.zeros((n,n),int); J=np.zeros((n,n)); OV=np.zeros((n,n))
    for i,a in enumerate(RL):
        for j,b in enumerate(RL):
            inter=int((active[a]&active[b]).sum()); union=int((active[a]|active[b]).sum())
            I[i,j]=inter; J[i,j]=inter/max(union,1); OV[i,j]=inter/max(min(sizes[a],sizes[b]),1)
    print("Active set sizes:",sizes)
    print("\nIntersection (# shared concepts):\n",pd.DataFrame(I,index=RL,columns=RL))
    print("\nJaccard overlap:\n",pd.DataFrame(J,index=RL,columns=RL).round(3))
    print("\nOverlap coefficient (shared / smaller set):\n",pd.DataFrame(OV,index=RL,columns=RL).round(3))
    lab=[NAME[r] for r in RL]; M=OV.copy(); np.fill_diagonal(M,np.nan)
    fig,ax=plt.subplots(figsize=(6.5,5.6))
    im=ax.imshow(M,cmap="YlGnBu",vmin=0,vmax=np.nanmax(M))
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(lab,rotation=30,ha="right"); ax.set_yticklabels(lab)
    for i in range(n):
        for j in range(n):
            if i!=j: ax.text(j,i,f"{OV[i,j]:.2f}",ha="center",va="center",fontsize=11,color="#222")
    fig.colorbar(im,label="Overlap Coefficient")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/cross_region_sharing.png",dpi=200,bbox_inches="tight")
    print("\nSAVED cross_region_sharing.png")
if __name__=="__main__": main()
