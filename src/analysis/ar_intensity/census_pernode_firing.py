"""
Node-level AR census in binary firing-rate terms (counts co-firing latents).
Reads pernode_{SAE}.npz. Per latent, per region: dP = P(active|AR node) - P(active|non-AR node).
Permutation null gives each latent its own spread; z=dP/std_null. Family-wise sig: z>99th pct of
null max-z across 4096. Each latent tested alone -> twins both count. Census = #regions significant ->
regional(1)/bi/tri/global, RELATIVE TO THESE 4 SAMPLED REGIONS. Also at dP cutoffs. Transfer=secondary.
"""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
SAE  = os.environ.get("SAE","plain_L8")
NPZ  = os.environ.get("PN", f"/scratch/euh7ys/climate_xai/concept_ivt/pernode_{SAE}.npz")
NSHUF= int(os.environ.get("NSHUF","200"))
OUT  = os.environ.get("OUT", f"/scratch/euh7ys/climate_xai/concept_ivt/census_fire_{SAE}.npz")
DPCUT= [0.05, 0.10, 0.20]

def main():
    d=np.load(NPZ, allow_pickle=True); RL=list(REGIONS)
    dPo={}; zo={}; sig={}; pAR={}; pNO={}; rng=np.random.default_rng(0)
    for r in RL:
        B=(np.asarray(d[f"{r}_X"])>0).astype(np.float32); y=d[f"{r}_y"].astype(np.float32)
        n1=float(y.sum()); n2=float(len(y)-n1); Sall=B.sum(0)
        Spos=y@B; p_ar=Spos/n1; p_no=(Sall-Spos)/n2; dobs=p_ar-p_no
        Y=np.empty((NSHUF,len(y)),np.float32)
        for s in range(NSHUF): Y[s]=rng.permutation(y)
        NS=Y@B; nulldP=NS/n1-(Sall-NS)/n2
        nstd=nulldP.std(0)+1e-12; z=dobs/nstd
        ceilz=float(np.percentile((nulldP/nstd).max(1),99)); sg=z>ceilz
        dPo[r]=dobs; zo[r]=z; sig[r]=sg; pAR[r]=p_ar; pNO[r]=p_no
        print(f"{r}: n_AR={int(n1)} maxdP={dobs.max():.3f}(latent {int(dobs.argmax())}) "
              f"z-ceil99={ceilz:.2f} #sig={int(sg.sum())} "
              + " ".join(f"#dP>{c}={int((dobs>c).sum())}" for c in DPCUT), flush=True)
        del B
    print("\n(reg/bi/tri/global relative to these 4 sampled regions only)")
    for lab, mask in [("significant", sig)]+[(f"dP>{c}", {r:dPo[r]>c for r in RL}) for c in DPCUT]:
        nreg=np.vstack([mask[r] for r in RL]).sum(0)
        c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"### CENSUS [{lab}]: active={int((nreg>=1).sum())} regional={c[1]} bi={c[2]} tri={c[3]} global={c[4]}")
    S=np.vstack([sig[r] for r in RL]); nreg=S.sum(0)
    print("\n### REGIONAL latents — significant in ONLY 1 of the 4 (id(dP, pAR v pNO)), top by dP:")
    for i,r in enumerate(RL):
        ro=np.where((nreg==1)&S[i])[0]; ro=ro[np.argsort(-dPo[r][ro])]
        print(f"  {r} ({len(ro)}): "+", ".join(f"{int(c)}({dPo[r][int(c)]:.2f},{pAR[r][int(c)]:.2f}v{pNO[r][int(c)]:.2f})" for c in ro[:12]))
    gl=np.where(nreg==4)[0]; gl=gl[np.argsort(-np.vstack([dPo[r] for r in RL]).min(0)[gl])]
    print(f"\n### GLOBAL latents — significant in all 4 ({len(gl)}): "+", ".join(str(int(c)) for c in gl[:20]))
    np.savez(OUT, dP=np.vstack([dPo[r] for r in RL]), z=np.vstack([zo[r] for r in RL]),
             sig=S, regions=np.array(RL), nreg=nreg)
    print("saved", OUT)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        cap=50000; sub={}
        for r in RL:
            B=(np.asarray(d[f"{r}_X"])>0).astype(np.float32); y=d[f"{r}_y"].astype(np.uint8)
            idx=np.random.default_rng(1).choice(len(y),min(cap,len(y)),replace=False); sub[r]=(B[idx],y[idx])
        clf={r:LogisticRegression(C=1.0,max_iter=300).fit(*sub[r]) for r in RL}
        print("\n### TRANSFER AUC (rows=train, cols=test) — secondary check:")
        print("        "+"".join(f"{r[:8]:>10}" for r in RL))
        for ra in RL:
            row=[roc_auc_score(sub[rb][1], clf[ra].predict_proba(sub[rb][0])[:,1]) for rb in RL]
            print(f"{ra[:8]:>8}"+"".join(f"{v:>10.3f}" for v in row))
    except Exception as e:
        print("transfer skipped:", e)

if __name__=="__main__": main()
