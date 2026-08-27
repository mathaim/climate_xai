"""
Per-latent node-level AR census + regional latent lists + transfer cross-check.

Reads pernode_{SAE}.npz. Per region, per latent: AUC = P(act@AR-node > act@nonAR-node), counting
co-firing latents (node-level labels do the confound control). Label-shuffle null -> family-wise
chance ceiling. A latent 'encodes AR in r' if AUC>bar.

DEFINITION OF "REGIONAL" (state this in the paper): counts are relative to the FOUR SAMPLED regions
only. 'regional' = AR-predictive in exactly ONE of the four (and not the other three); it is NOT a
claim the latent predicts AR nowhere else on Earth. bi/tri/global = predictive in 2/3/4 of the four.

Reported at the chance ceiling AND at effect-size bars (at ~1e5 samples 'beats chance' ~ AUC>0.505,
so the ceiling flags weak-but-real effects; the effect-size bars show the strong concepts). Prints
AND saves the regional latent IDs (ranked by AUC) so the concepts themselves are the output.
Rows shuffled per region so the TopK zero tie-block breaks randomly, not by class order.
"""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
SAE  = os.environ.get("SAE","plain_L8")
NPZ  = os.environ.get("PN", f"/scratch/euh7ys/climate_xai/concept_ivt/pernode_{SAE}.npz")
NSHUF= int(os.environ.get("NSHUF","40"))
OUT  = os.environ.get("OUT", f"/scratch/euh7ys/climate_xai/concept_ivt/census_{SAE}.npz")
TAUS = [0.55, 0.60, 0.70]

def auc_cols(X, y, cs=512):
    n1=int(y.sum()); n2=len(y)-n1; C=X.shape[1]; out=np.zeros(C); pos=y.astype(bool)
    for i in range(0,C,cs):
        R=np.argsort(np.argsort(X[:,i:i+cs].astype(np.float32),0),0).astype(np.float64)+1.0
        out[i:i+cs]=(R[pos].sum(0)-n1*(n1+1)/2)/max(n1*n2,1)
    return out

def main():
    d=np.load(NPZ, allow_pickle=True); RL=list(REGIONS); A={}; ceil={}; data={}
    rng0=np.random.default_rng(0)
    for r in RL:
        X=d[f"{r}_X"].astype(np.float32); y=d[f"{r}_y"].astype(np.uint8)
        p=rng0.permutation(len(y)); X=X[p]; y=y[p]; data[r]=(X,y); A[r]=auc_cols(X,y)
        rng=np.random.default_rng(1); nm=[auc_cols(X,rng.permutation(y)).max() for _ in range(NSHUF)]
        ceil[r]=float(np.percentile(nm,99))
        print(f"{r}: n_AR={int(y.sum())} n_nonAR={int((y==0).sum())} maxAUC={A[r].max():.3f} "
              f"ceil99={ceil[r]:.3f} | #>ceil={int((A[r]>ceil[r]).sum())} "
              + " ".join(f"#>{t}={int((A[r]>t).sum())}" for t in TAUS), flush=True)

    print("\n(NOTE: 'regional/bi/tri/global' are relative to these 4 sampled regions only.)")
    Amat=np.vstack([A[r] for r in RL])
    for label,bars in [("chance-ceiling",{r:ceil[r] for r in RL})]+[(f"AUC>{t}",{r:t for r in RL}) for t in TAUS]:
        nreg=np.vstack([A[r]>bars[r] for r in RL]).sum(0)
        c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"### CENSUS [{label}]: active={int((nreg>=1).sum())} regional={c[1]} bi={c[2]} tri={c[3]} global={c[4]}")

    Mc=np.vstack([A[r]>ceil[r] for r in RL]); nregc=Mc.sum(0)
    print("\n### REGIONAL latents — AR-predictive in ONLY 1 of the 4 sampled regions (top by AUC):")
    for i,r in enumerate(RL):
        ro=np.where((nregc==1)&Mc[i])[0]; ro=ro[np.argsort(-A[r][ro])]
        print(f"  {r} ({len(ro)}): " + ", ".join(f"{int(c)}({A[r][int(c)]:.2f})" for c in ro[:15]))
    gl=np.where(nregc==4)[0]; gl=gl[np.argsort(-Amat.min(0)[gl])]
    print(f"\n### GLOBAL latents — predictive in all 4 sampled regions (top by weakest-region AUC): " +
          ", ".join(f"{int(c)}({Amat.min(0)[int(c)]:.2f})" for c in gl[:15]))
    np.savez(OUT, auc=Amat, ceil=np.array([ceil[r] for r in RL]),
             regions=np.array(RL), nreg=nregc)
    print("saved", OUT)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        clf={}; sc={}
        for r in RL:
            X,y=data[r]; s=StandardScaler().fit(X)
            clf[r]=LogisticRegression(C=1.0,max_iter=300).fit(s.transform(X),y); sc[r]=s
        print("\n### TRANSFER AUC (rows=train, cols=test) — high off-diagonal = shared AR direction:")
        print("        "+"".join(f"{r[:8]:>10}" for r in RL))
        for ra in RL:
            row=[roc_auc_score(data[rb][1], clf[ra].predict_proba(sc[ra].transform(data[rb][0]))[:,1]) for rb in RL]
            print(f"{ra[:8]:>8}"+"".join(f"{v:>10.3f}" for v in row))
    except Exception as e:
        print("transfer skipped:", e)

if __name__=="__main__": main()
