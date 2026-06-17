import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.analysis.ar_intensity.sae_features import SAES
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity._load import load
def main():
    bacc=make_scorer(balanced_accuracy_score); cv=StratifiedKFold(5,shuffle=True,random_state=0)
    print(f"{'SAE':10} | "+"  ".join(f"{r[:8]:>8}" for r in REGIONS)+" |  pooled   (chance balanced=0.25)")
    for name in SAES:
        F,md=load(name); y=md.intensity_bin.to_numpy(); reg=md.region.to_numpy(); accs=[]
        for r in REGIONS:
            mk=reg==r
            clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            accs.append(cross_val_score(clf,F[mk],y[mk],cv=cv,scoring=bacc).mean())
        clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pooled=cross_val_score(clf,F,y,cv=cv,scoring=bacc).mean()
        print(f"{name:10} | "+"  ".join(f"{a:8.3f}" for a in accs)+f" |  {pooled:.3f}",flush=True)
if __name__=="__main__": main()
