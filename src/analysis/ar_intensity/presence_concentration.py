"""Test A on AR PRESENCE: held-out AR-vs-non-AR decode accuracy vs # latents, Plain vs Matryoshka."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity._load import load
RES="results/ar_intensity"; OUT=f"{RES}/baseline"; VAR="binary"
KS=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096]; SEEDS=[0,1,2]
def pbcorr(X,y):
    Xc=X-X.mean(0,keepdims=True); yc=y-y.mean()
    return (Xc.T@yc)/(np.sqrt((Xc*Xc).sum(0)*(yc@yc))+1e-12)
def main():
    rows=[]
    for arch,name in [("plain","plain_L8"),("matry","matry_L8")]:
        Far,md=load(name,f"region_{VAR}"); reg=md.region.to_numpy()
        no=np.load(f"{OUT}/nonar_pertimestep_{name}.npz")
        for r in REGIONS:
            Xar=Far[reg==r]; Xno=no[f"{r}_{VAR}"]
            n=min(len(Xar),len(Xno)); rng=np.random.default_rng(0)
            Xar=Xar[rng.choice(len(Xar),n,replace=False)]; Xno=Xno[rng.choice(len(Xno),n,replace=False)]
            X=np.vstack([Xar,Xno]); y=np.r_[np.ones(n),np.zeros(n)]
            for sd in SEEDS:
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.5,random_state=sd,stratify=y)
                order=np.argsort(-np.abs(pbcorr(Xtr.astype(np.float64),ytr)))
                for k in KS:
                    top=order[:k]; clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=400)); clf.fit(Xtr[:,top],ytr)
                    rows.append(dict(arch=arch,region=r,seed=sd,k=k,bacc=balanced_accuracy_score(yte,clf.predict(Xte[:,top]))))
            print(name,r,"done",flush=True)
    df=pd.DataFrame(rows); df.to_csv(f"{RES}/presence_concentration.csv",index=False)
    print(df.groupby(["arch","k"]).bacc.mean().unstack(0).round(3)); print("SAVED presence_concentration.csv")
if __name__=="__main__": main()
