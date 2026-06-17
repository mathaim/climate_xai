import numpy as np
from src.analysis.ar_intensity.sae_features import SAES, load_sae
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.binning import BINS
from src.analysis.ar_intensity._load import load
RANK={b:i for i,b in enumerate(BINS)}

def get_groups(m):
    # try to read nested-dict sizes from the model; fall back to known prefixes
    for a in ("group_sizes","matryoshka_group_sizes","prefixes","sizes","group_sizes_"):
        if hasattr(m,a):
            v=[int(x) for x in getattr(m,a)]
            print(f"  [groups from model.{a}] = {v}")
            return v
    print("  [groups: model attr not found, using prefixes 256/512/1024/2048/4096]")
    return [256,512,1024,2048,4096]

def main():
    m0=load_sae("matry_L0","cpu")[0]
    pre=get_groups(m0)                       # cumulative prefix sizes
    bounds=[0]+pre
    groups=[(bounds[i],bounds[i+1]) for i in range(len(pre))]
    for name in ["matry_L0","matry_L8","matry_L15"]:
        F,md=load(name)
        pct=np.zeros((len(groups),len(REGIONS))); mr=np.zeros_like(pct)
        for ri,r in enumerate(REGIONS):
            m=(md.region==r).to_numpy(); X=F[m].astype(np.float64)
            rk=md.loc[m,"intensity_bin"].map(RANK).to_numpy(float)
            Xc=X-X.mean(0,keepdims=True); rc=rk-rk.mean()
            corr=(Xc.T@rc)/(np.sqrt((Xc*Xc).sum(0)*(rc@rc))+1e-12)
            a=np.abs(corr)
            for gi,(lo,hi) in enumerate(groups):
                seg=a[lo:hi]; pct[gi,ri]=100*(seg>0.2).mean(); mr[gi,ri]=seg.mean()
        print(f"\n=== {name}: intensity tracking by nested group (mean across 4 regions) ===")
        print(f"{'group':>16} {'size':>5} {'%track':>8} {'mean|r|':>9}")
        for gi,(lo,hi) in enumerate(groups):
            print(f"  G{gi} [{lo:>4}:{hi:<4}] {hi-lo:>5} {pct[gi].mean():7.1f}% {mr[gi].mean():9.3f}")
if __name__=="__main__": main()
