"""Do outer (child) latents fire with core (parent) latents? P(core|outer) + lift, Matryoshka vs Plain."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae
COF="/scratch/euh7ys/climate_xai/cofire"
def load_cf(name):
    d=np.load(f"{COF}/cofire_{name}.npz"); return d["cofire"].astype(np.float64),d["fire"].astype(np.float64),float(d["nodes"][0])
def main():
    pre=[int(x) for x in load_sae("matry_L0","cpu")[0].group_sizes]
    core=np.arange(0,pre[1]); outer=np.arange(pre[3],pre[4])
    rows=[]
    for name in ["matry_L8","plain_L8"]:
        C,f,N=load_cf(name); P=f/N
        valid=f[outer]>50
        Pcond=C[np.ix_(core,outer)]/np.maximum(f[outer][None,:],1.0)
        bestP=Pcond.max(0); besti=core[Pcond.argmax(0)]; lift=bestP/np.maximum(P[besti],1e-12)
        rows.append(dict(sae=name,mean_P_parent_given_child=float(np.mean(bestP[valid])),
                         median_lift=float(np.median(lift[valid])),
                         frac_children_with_strong_parent=float(np.mean(bestP[valid]>0.5))))
        np.savez(f"{COF}/cond_{name}.npz",bestP=bestP,lift=lift,valid=valid)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for name,lab,col in [("matry_L8","Matryoshka SAE","#2b6cb0"),("plain_L8","Plain SAE (index)","#c0392b")]:
        d=np.load(f"{COF}/cond_{name}.npz"); v=d["valid"]
        ax[0].hist(d["bestP"][v],bins=40,alpha=.5,color=col,density=True,label=lab)
        ax[1].hist(np.clip(d["lift"][v],0,10),bins=40,alpha=.5,color=col,density=True,label=lab)
    ax[0].set_xlabel("P(core parent fires | outer child fires)"); ax[0].set_ylabel("density"); ax[0].legend()
    ax[1].set_xlabel("Lift = P(parent|child) / P(parent)"); ax[1].axvline(1,ls="--",c="#888"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{COF}/conditional_firing.png",dpi=180,bbox_inches="tight")
    print("SAVED",f"{COF}/conditional_firing.png + cond_*.npz")
if __name__=="__main__": main()
