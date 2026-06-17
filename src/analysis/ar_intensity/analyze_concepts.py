import numpy as np
from src.analysis.ar_intensity.sae_features import SAES
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.binning import BINS
from src.analysis.ar_intensity._load import load
RANK={b:i for i,b in enumerate(BINS)}
def main():
    for name in SAES:
        F,md=load(name); print(f"=== {name} ===")
        for r in REGIONS:
            m=(md.region==r).to_numpy()
            X=F[m].astype(np.float64)
            rk=md.loc[m,"intensity_bin"].map(RANK).to_numpy(float)
            Xc=X-X.mean(0,keepdims=True); rc=rk-rk.mean()
            num=Xc.T@rc; den=np.sqrt((Xc*Xc).sum(0)*(rc@rc))+1e-12
            corr=num/den
            ntr=int((np.abs(corr)>0.2).sum()); top=np.argsort(-corr)[:5]
            print(f"  {r:13} n={m.sum():5d}: tracking(|r|>0.2)={ntr:4d} | top+: "+
                  " ".join(f"{int(c)}({corr[c]:+.2f})" for c in top))
if __name__=="__main__": main()
