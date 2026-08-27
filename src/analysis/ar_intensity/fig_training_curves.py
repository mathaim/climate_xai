"""Appendix training curves: reconstruction loss vs fraction of training, one panel per architecture."""
import json, math, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
matplotlib.rcParams.update({"font.size":14,"axes.titlesize":17,"axes.labelsize":16,
                            "xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":14,"legend.title_fontsize":14})
PLAIN = {"Layer 0":"/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer00",
         "Layer 8":"/scratch/euh7ys/climate_xai/checkpoints/plain_layer8",
         "Layer 15":"/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer15"}
MATRY = {"Layer 0":"/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer00",
         "Layer 8":"/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer08",
         "Layer 15":"/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer15"}
COL = {"Layer 0":"#2980b9","Layer 8":"#c0392b","Layer 15":"#27ae60"}
def rollmed(y,w=15):
    y=np.asarray(y,float); n=len(y); h=w//2
    return np.array([np.median(y[max(0,i-h):min(n,i+h+1)]) for i in range(n)])
def sci(v):
    if v<=0: return "0"
    e=int(math.floor(math.log10(v))); m=v/10.0**e
    if abs(m-1)<0.05: return rf"$10^{{{e}}}$"
    ms=f"{m:.0f}" if abs(m-round(m))<0.05 else f"{m:.1f}"
    return rf"${ms}\times10^{{{e}}}$"
fig,(axP,axM)=plt.subplots(1,2,figsize=(11,4.8))
for lab,p in PLAIN.items():
    rows=[json.loads(l) for l in open(f"{p}/training_log.jsonl")]
    seen={}
    for r in rows: seen[r["epoch"]]=r
    ep=sorted(seen); emax=max(ep)
    axP.plot([e/emax for e in ep],[seen[e]["recon_loss"] for e in ep],"-o",ms=4,color=COL[lab],label=lab)
mmin,mmax=np.inf,-np.inf
for lab,p in MATRY.items():
    rows=[json.loads(l) for l in open(f"{p}/training_log.jsonl")]
    st=[i for i,r in enumerate(rows) if r["step"]==500]
    if st: rows=rows[st[-1]:]
    step=np.array([r["step"] for r in rows],float); loss=np.array([r["loss"] for r in rows],float)
    x=step/step.max(); med=rollmed(loss)
    axM.plot(x,loss,"-",lw=0.5,alpha=0.22,color=COL[lab])
    axM.plot(x,med,"-",lw=1.8,color=COL[lab],label=lab)
    mmin=min(mmin,float(med.min())); mmax=max(mmax,float(med.max()))
axP.set_ylim(1.0e-4,4.5e-4); axM.set_ylim(50,mmax*1.30)
STEP={"A":1.0e-4,"B":50.0}
for ax,t,letter in [(axP,"Standard SAE","A"),(axM,"Matryoshka SAE","B")]:
    ax.set_xlabel("Fraction of Training"); ax.grid(alpha=.3,which="major")
    ax.set_title(t); ax.set_ylabel("Training Reconstruction Loss")
    ax.text(0.0,1.05,letter,transform=ax.transAxes,fontsize=19,fontweight="bold",va="bottom",ha="left")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(STEP[letter]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: sci(v)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
h,l=axP.get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=3,title="Depth",frameon=False,bbox_to_anchor=(0.5,0.0))
fig.align_ylabels([axP,axM])
fig.tight_layout(rect=[0,0.11,1,1],w_pad=3.0)
fig.savefig("/scratch/euh7ys/climate_xai/plots/training_curves.png",dpi=170,bbox_inches="tight")
print("saved training_curves.png  matry ylim top=%.1f"%(mmax*1.30))
