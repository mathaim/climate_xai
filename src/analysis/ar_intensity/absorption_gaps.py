"""Test B: absorption gaps. Does the main 'AR present' latent fail more on high-intensity ARs (child subset)?"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.binning import BINS
from src.analysis.ar_intensity._load import load
RES="results/ar_intensity"; OUT=f"{RES}/baseline"; VAR="binary"; SEEDS=[0,1,2]
def pbcorr(X,y):
    Xc=X-X.mean(0,keepdims=True); yc=y-y.mean()
    return (Xc.T@yc)/(np.sqrt((Xc*Xc).sum(0)*(yc@yc))+1e-12)
def best_thr(v,y):
    cands=np.unique(np.percentile(v,np.linspace(2,98,40))); best,bb=cands[0],-1
    for t in cands:
        ba=balanced_accuracy_score(y,(v>t).astype(int))
        if ba>bb: bb,best=ba,t
    return best
def main():
    rows=[]
    for arch,name in [("plain","plain_L8"),("matry","matry_L8")]:
        F,md=load(name,f"region_{VAR}"); reg=md.region.to_numpy(); bins=md.intensity_bin.to_numpy()
        no=np.load(f"{OUT}/nonar_pertimestep_{name}.npz")
        for r in REGIONS:
            Xar=F[reg==r]; bar=bins[reg==r]; Xno=no[f"{r}_{VAR}"]; nar=len(Xar)
            X=np.vstack([Xar,Xno]); y=np.r_[np.ones(nar),np.zeros(len(Xno))]
            binlab=np.array(list(bar)+[None]*len(Xno),dtype=object)
            for sd in SEEDS:
                idx=np.arange(len(X)); tr,te=train_test_split(idx,test_size=0.5,random_state=sd,stratify=y)
                main=int(np.argmax(np.abs(pbcorr(X[tr].astype(np.float64),y[tr]))))
                thr=best_thr(X[tr,main],y[tr]); te_ar=te[y[te]==1]
                for b in BINS:
                    sel=te_ar[binlab[te_ar]==b]
                    if len(sel): rows.append(dict(arch=arch,region=r,seed=sd,bin=b,miss=float(np.mean(X[sel,main]<=thr))))
            print(name,r,"done",flush=True)
    df=pd.DataFrame(rows); df.to_csv(f"{RES}/absorption_gaps.csv",index=False)
    print("=== Main 'AR-present' latent MISS RATE by intensity bin (mean over regions+seeds) ===")
    print(df.groupby(["arch","bin"]).miss.mean().unstack(0).reindex(BINS).round(3))
    for arch in ["plain","matry"]:
        d=df[df.arch==arch].groupby("bin").miss.mean()
        print(f"  {arch}: absorption score (top10 miss - bottom10 miss) = {d['top10']-d['bottom10']:+.3f}")
    print("SAVED absorption_gaps.csv")
if __name__=="__main__": main()
