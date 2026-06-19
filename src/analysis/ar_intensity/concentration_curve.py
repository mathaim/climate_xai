"""Test A: intensity-signal concentration (held-out decode accuracy vs # latents), Plain vs Matryoshka."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.binning import BINS
from src.analysis.ar_intensity._load import load
RES="results/ar_intensity"; RANK={b:i for i,b in enumerate(BINS)}; VAR="region_magnitude"
KS=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096]; SEEDS=[0,1,2]; CAP=12000
def corr(X,rk):
    Xc=X-X.mean(0,keepdims=True); rc=rk-rk.mean()
    return (Xc.T@rc)/(np.sqrt((Xc*Xc).sum(0)*(rc@rc))+1e-12)
def main():
    rows=[]
    for arch,name in [("plain","plain_L8"),("matry","matry_L8")]:
        F,md=load(name,VAR); y=md.intensity_bin.to_numpy(); reg=md.region.to_numpy(); rkall=md.intensity_bin.map(RANK).to_numpy(float)
        for r in REGIONS:
            mk=reg==r; X=F[mk]; yr=y[mk]; rk=rkall[mk]
            if len(yr)>CAP:
                idx=np.random.default_rng(0).choice(len(yr),CAP,replace=False); X,yr,rk=X[idx],yr[idx],rk[idx]
            for sd in SEEDS:
                Xtr,Xte,ytr,yte,rktr,_=train_test_split(X,yr,rk,test_size=0.5,random_state=sd,stratify=yr)
                order=np.argsort(-np.abs(corr(Xtr.astype(np.float64),rktr)))
                for k in KS:
                    top=order[:k]
                    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=400)); clf.fit(Xtr[:,top],ytr)
                    rows.append(dict(arch=arch,region=r,seed=sd,k=k,bacc=balanced_accuracy_score(yte,clf.predict(Xte[:,top]))))
            print(name,r,"done",flush=True)
    df=pd.DataFrame(rows); df.to_csv(f"{RES}/concentration_curve.csv",index=False)
    print(df.groupby(["arch","k"]).bacc.mean().unstack(0).round(3))
    print("SAVED concentration_curve.csv")
if __name__=="__main__": main()
