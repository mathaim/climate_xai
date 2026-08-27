"""Exploratory node-level firing-rate RATIO. Per concept f, per region:
  rate_AR = P(f active | node masked AR);  rate_no = P(f active | node not AR)
  ratio = rate_AR / rate_no   (>1 => active more at AR nodes). Normalized by the AR:non-AR base rate.
Descriptive exploration, no causal claim. Reads pertimestep_{SAE}."""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
SAE=os.environ.get("SAE","plain_L8"); CH=256
DIR=f"/scratch/euh7ys/climate_xai/concept_ivt/pertimestep_{SAE}"
def main():
    RL=list(REGIONS); RAT={}
    for r in RL:
        d=np.load(f"{DIR}/{r}.npz"); Apack=d["A"]; M=d["M"].astype(np.float32)
        N_AR=float(M.sum()); N_no=float(M.size)-N_AR
        cAR=np.zeros(4096); cALL=np.zeros(4096)
        for c0 in range(0,4096,CH):
            B=np.unpackbits(Apack[:,:,c0//8:(c0+CH)//8],axis=-1).astype(np.float32)
            cAR[c0:c0+CH]=np.einsum('tnf,tn->f',B,M); cALL[c0:c0+CH]=B.sum((0,1))
        rAR=cAR/N_AR; rno=(cALL-cAR)/N_no; RAT[r]=rAR/(rno+1e-9)
        print(f"{r}: N_AR={int(N_AR)} N_no={int(N_no)} maxRatio={RAT[r].max():.1f}(lat {int(RAT[r].argmax())})",flush=True)
    for th in (1.5,2,3):
        nreg=np.vstack([RAT[r]>th for r in RL]).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  ratio>{th}: active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    g=np.where(np.vstack([RAT[r]>2 for r in RL]).sum(0)==4)[0]; print(f"  global(ratio>2 all4): {g.tolist()}")
    for r in RL:
        idx=np.where(RAT[r]>2)[0]; idx=idx[np.argsort(-RAT[r][idx])]
        print(f"    {r}: "+", ".join(f"{int(x)}({RAT[r][int(x)]:.1f})" for x in idx[:8]))
    np.savez(f"/scratch/euh7ys/climate_xai/concept_ivt/ratio_{SAE}.npz",ratio=np.vstack([RAT[r] for r in RL]),regions=np.array(RL))
if __name__=="__main__": main()
