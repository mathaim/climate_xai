"""Direct-mask census. Region-AR = ANY AR in region (M[t].any()); no-AR = none.
Per node i, latent f: P(fire|region-AR) - P(fire|region-no-AR). Delta(f)=mean over region nodes.
Region-level AR condition (strong, clean) + per-node firing (GNN response). pertimestep_{SAE}."""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
SAE=os.environ.get("SAE","plain_L8"); CH=256
DIR=f"/scratch/euh7ys/climate_xai/concept_ivt/pertimestep_{SAE}"
def main():
    RL=list(REGIONS); DP={}
    for r in RL:
        d=np.load(f"{DIR}/{r}.npz"); Apack=d["A"]; M=d["M"].astype(bool)
        regAR=M.any(1); nAR=int(regAR.sum()); nNO=int((~regAR).sum()); T,n=M.shape
        arf=regAR.astype(np.float32); delta=np.zeros(4096)
        for c0 in range(0,4096,CH):
            B=np.unpackbits(Apack[:,:,c0//8:(c0+CH)//8],axis=-1).astype(np.float32)  # (T,n,CH)
            sAR=np.einsum('tnf,t->nf',B,arf); sALL=B.sum(0)
            muAR=sAR/max(nAR,1); muNO=(sALL-sAR)/max(nNO,1)
            delta[c0:c0+CH]=(muAR-muNO).mean(0)
        DP[r]=delta
        print(f"{r}: nodes={n} nAR={nAR} nNO={nNO} maxDelta={delta.max():.3f}(lat {int(delta.argmax())})",flush=True)
    for th in (0.02,0.05,0.1,0.2):
        nreg=np.vstack([DP[r]>th for r in RL]).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  Delta>{th}: active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    g=np.where(np.vstack([DP[r]>0.05 for r in RL]).sum(0)==4)[0]; print(f"  global(Delta>0.05 all4): {g.tolist()}")
    np.savez(f"/scratch/euh7ys/climate_xai/concept_ivt/anyAR_{SAE}.npz",delta=np.vstack([DP[r] for r in RL]),regions=np.array(RL))
if __name__=="__main__": main()
