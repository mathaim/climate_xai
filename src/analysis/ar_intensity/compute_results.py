"""Stage-5 result tables over the 6 firing variants -> tidy CSVs + per-concept corr arrays."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.analysis.ar_intensity.sae_features import SAES, load_sae
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.binning import BINS
from src.analysis.ar_intensity._load import load
RES="results/ar_intensity"; RANK={b:i for i,b in enumerate(BINS)}
VARIANTS=[f"{s}_{d}" for s in ["region","global"] for d in ["binary","magnitude","top10"]]
def corr(X,rk):
    Xc=X-X.mean(0,keepdims=True); rc=rk-rk.mean()
    return (Xc.T@rc)/(np.sqrt((Xc*Xc).sum(0)*(rc@rc))+1e-12)
def main():
    os.makedirs(f"{RES}/plots",exist_ok=True); os.makedirs(f"{RES}/corr",exist_ok=True)
    bacc=make_scorer(balanced_accuracy_score); cv=StratifiedKFold(5,shuffle=True,random_state=0)
    pre=[int(x) for x in load_sae("matry_L0","cpu")[0].group_sizes]; gb=[0]+pre
    groups=[(gb[i],gb[i+1]) for i in range(len(pre))]
    dec=[]; csum=[]; grp=[]; tun=[]
    for name,cfg in SAES.items():
        arch,layer=cfg["arch"],cfg["layer"]
        for v in VARIANTS:
            F,md=load(name,v); y=md.intensity_bin.to_numpy(); reg=md.region.to_numpy()
            rkall=md.intensity_bin.map(RANK).to_numpy(float)
            for r in list(REGIONS)+["pooled"]:
                mk=slice(None) if r=="pooled" else (reg==r)
                clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
                a=cross_val_score(clf,F[mk],y[mk],cv=cv,scoring=bacc).mean()
                dec.append(dict(sae=name,arch=arch,layer=layer,variant=v,region=r,balanced_acc=a))
            for r in REGIONS:
                m=(reg==r); X=F[m].astype(np.float64); rk=rkall[m]; sub=md.loc[m,"intensity_bin"].to_numpy()
                cc=corr(X,rk); np.save(f"{RES}/corr/{name}_{v}_{r}.npy",cc.astype(np.float32))
                csum.append(dict(sae=name,arch=arch,layer=layer,variant=v,region=r,
                                 mean_abs_r=float(np.abs(cc).mean()),frac_strong=float((np.abs(cc)>0.2).mean())))
                prof=np.zeros((4096,len(BINS)))
                for bi,b in enumerate(BINS):
                    s=sub==b
                    if s.any(): prof[:,bi]=X[s].mean(0)
                tot=prof.sum(1); active=tot>0
                pref=prof.argmax(1); extreme=prof[:,-1]/(prof.mean(1)+1e-12)
                d=pd.DataFrame({"concept":np.where(active)[0],"r":cc[active],
                                "preferred_bin":pref[active],"extreme_enrichment":extreme[active]})
                for bi in range(len(BINS)): d[f"b{bi}"]=prof[active,bi]
                d["sae"]=name; d["arch"]=arch; d["layer"]=layer; d["variant"]=v; d["region"]=r
                tun.append(d)
                if arch=="matry":
                    for gi,(lo,hi) in enumerate(groups):
                        seg=np.abs(cc[lo:hi])
                        grp.append(dict(sae=name,layer=layer,variant=v,region=r,group=gi,size=hi-lo,
                                        mean_abs_r=float(seg.mean()),pct_strong=float((seg>0.2).mean()*100)))
            print(name,v,"done",flush=True)
    pd.DataFrame(dec).to_csv(f"{RES}/decodability.csv",index=False)
    pd.DataFrame(csum).to_csv(f"{RES}/corr_summary.csv",index=False)
    pd.DataFrame(grp).to_csv(f"{RES}/matry_group.csv",index=False)
    pd.concat(tun,ignore_index=True).to_csv(f"{RES}/concept_tuning.csv",index=False)
    print("RESULTS SAVED",flush=True)
if __name__=="__main__": main()
