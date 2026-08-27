"""Per-node conditional AR census (self-controlled). For node i, latent f:
   mu_AR(f,i)=firing rate of f at node i when an AR is over node i;  mu_no(f,i)=when not.
Delta(f) = mean over region nodes of (mu_AR - mu_no). Each node is its own control, so spatial
climatology cancels -> isolates the DYNAMIC AR response, using non-AR frames. Census by Delta>thr.
Reads pertimestep_{SAE}/{region}.npz."""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
SAE=os.environ.get("SAE","plain_L8"); CH=256
DIR=f"/scratch/euh7ys/climate_xai/concept_ivt/pertimestep_{SAE}"
def main():
    RL=list(REGIONS); DELTA={}
    for r in RL:
        d=np.load(f"{DIR}/{r}.npz"); Apack=d["A"]; M=d["M"].astype(np.float32)   # A(T,n,512) M(T,n)
        T,n=M.shape; cAR=M.sum(0); cNO=T-cAR; valid=(cAR>0)&(cNO>0)
        delta=np.zeros(4096)
        for c0 in range(0,4096,CH):
            B=np.unpackbits(Apack[:,:,c0//8:(c0+CH)//8],axis=-1).astype(np.float32)  # (T,n,CH)
            sAR=np.einsum('tnf,tn->nf',B,M); sALL=B.sum(0)
            muAR=sAR/np.maximum(cAR[:,None],1); muNO=(sALL-sAR)/np.maximum(cNO[:,None],1)
            delta[c0:c0+CH]=(muAR-muNO)[valid].mean(0)
        DELTA[r]=delta
        print(f"{r}: nodes={n} T={T} valid={int(valid.sum())} maxDelta={delta.max():.3f}(lat {int(delta.argmax())})",flush=True)
    print()
    for th in (0.005,0.01,0.02,0.05):
        nreg=np.vstack([DELTA[r]>th for r in RL]).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"  Delta>{th}: active={int((nreg>=1).sum())} reg={c[1]} bi={c[2]} tri={c[3]} glob={c[4]}")
    g=np.where(np.vstack([DELTA[r]>0.01 for r in RL]).sum(0)==4)[0]
    print(f"  global (Delta>0.01 all 4): {g.tolist()[:25]}")
    for r in RL:
        idx=np.where(DELTA[r]>0.01)[0]; idx=idx[np.argsort(-DELTA[r][idx])]
        print(f"    {r}: "+", ".join(f"{int(x)}({DELTA[r][int(x)]:.3f})" for x in idx[:8]))
    np.savez(f"/scratch/euh7ys/climate_xai/concept_ivt/pernode_cond_{SAE}.npz",
             delta=np.vstack([DELTA[r] for r in RL]),regions=np.array(RL))
if __name__=="__main__": main()
