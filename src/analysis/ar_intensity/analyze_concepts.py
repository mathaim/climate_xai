import numpy as np, pandas as pd
from src.analysis.ar_intensity.sae_features import SAES
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.binning import BINS
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
RANK={b:i for i,b in enumerate(BINS)}
def main():
    for name in SAES:
        F=np.load(f"{OUTDIR}/{name}_features.npy"); md=pd.read_parquet(f"{OUTDIR}/{name}_meta.parquet")
        print(f"\n=== {name} ===")
        for r in REGIONS:
            m=(md.region==r).values; X=F[m]
            rk=md.loc[m,"intensity_bin"].map(RANK).values.astype(float)
            Xc=X-X.mean(0); rc=rk-rk.mean()
            corr=(Xc*rc[:,None]).sum(0)/(np.sqrt((Xc**2).sum(0)*(rc**2).sum())+1e-12)
            ntr=int((np.abs(corr)>0.2).sum()); top=np.argsort(-corr)[:5]
            print(f"  {r:13}: tracking(|r|>0.2)={ntr:4d} | top+: "+
                  " ".join(f"{int(c)}({corr[c]:+.2f})" for c in top))
if __name__=="__main__": main()
