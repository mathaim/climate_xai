"""
Node-level AR census via MUTUAL INFORMATION (a KL divergence) + odds-vs-noise significance.
Reads pernode_{SAE}.npz. Per latent, 2x2 table {active/inactive}x{AR/non-AR}:
  MI(latent;AR) bits = KL( P(active,AR) || P(active)P(AR) ) = how much the latent tells you about AR.
NOISE = null that latent activity is INDEPENDENT of AR (p_AR=p_no). Permutation (shuffle AR labels) ->
by-chance MI distribution; family-wise ceiling = 99th pct of max-MI over 4096. A latent 'captures AR'
in r if MI>ceiling AND AR-excited (p_AR>p_no). Counts twins. MI credits rare-but-specific latents dP misses.
Census -> regional(1)/bi/tri/global relative to the 4 sampled regions, at the noise ceiling and MI cutoffs.
"""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
SAE=os.environ.get("SAE","plain_L8")
NPZ=os.environ.get("PN", f"/scratch/euh7ys/climate_xai/concept_ivt/pernode_{SAE}.npz")
NSHUF=int(os.environ.get("NSHUF","200"))
OUT=os.environ.get("OUT", f"/scratch/euh7ys/climate_xai/concept_ivt/census_mi_{SAE}.npz")
MICUT=[0.002,0.005,0.01]

def mi_bits(a1, Atot, n1, n0, N):
    Nin=N-Atot
    cells=[a1, n1-a1, Atot-a1, n0-(Atot-a1)]      # AR&act, AR&inact, nonAR&act, nonAR&inact
    rows =[n1, n1, n0, n0]; cols=[Atot, Nin, Atot, Nin]
    mi=np.zeros(np.broadcast(a1,Atot).shape)
    for c,rw,cl in zip(cells,rows,cols):
        exp=np.asarray(rw)*np.asarray(cl)/N
        with np.errstate(divide="ignore",invalid="ignore"):
            t=c*np.log2(np.where(c>0,c/exp,1.0))
        mi=mi+np.where(c>0,t,0.0)
    return mi/N

def main():
    d=np.load(NPZ, allow_pickle=True); RL=list(REGIONS)
    MI={}; dPo={}; pAR={}; pNO={}; ORr={}; sig={}; rng=np.random.default_rng(0)
    for r in RL:
        B=(np.asarray(d[f"{r}_X"])>0).astype(np.float32); y=d[f"{r}_y"].astype(np.float32)
        N=len(y); n1=float(y.sum()); n0=N-n1; Atot=B.sum(0); a1=y@B
        mi=mi_bits(a1,Atot,n1,n0,N); p_ar=a1/n1; p_no=(Atot-a1)/n0
        a0=Atot-a1; OR=((a1+.5)*(n0-a0+.5))/((n1-a1+.5)*(a0+.5))
        Y=np.empty((NSHUF,N),np.float32)
        for s in range(NSHUF): Y[s]=rng.permutation(y)
        minull=mi_bits(Y@B, Atot[None,:], n1, n0, N)
        cz=float(np.percentile(minull.max(1),99)); noise=float(minull.mean())
        exc=p_ar>p_no; sg=(mi>cz)&exc
        MI[r]=mi; dPo[r]=p_ar-p_no; pAR[r]=p_ar; pNO[r]=p_no; ORr[r]=OR; sig[r]=sg
        print(f"{r}: maxMI={mi.max():.4f}b(latent {int(mi.argmax())}) noise~{noise:.1e}b ceil99={cz:.4f}b "
              f"#sig={int(sg.sum())} "+" ".join(f"#MI>{c}={int(((mi>c)&exc).sum())}" for c in MICUT), flush=True)
        del B
    print("\n(reg/bi/tri/global relative to these 4 sampled regions only)")
    for lab,mask in [("MI>noise-ceiling",sig)]+[(f"MI>{c}",{r:(MI[r]>c)&(dPo[r]>0) for r in RL}) for c in MICUT]:
        nreg=np.vstack([mask[r] for r in RL]).sum(0); c={k:int((nreg==k).sum()) for k in (1,2,3,4)}
        print(f"### CENSUS [{lab}]: active={int((nreg>=1).sum())} regional={c[1]} bi={c[2]} tri={c[3]} global={c[4]}")
    S=np.vstack([sig[r] for r in RL]); nreg=S.sum(0)
    print("\n### REGIONAL latents — significant in ONLY 1 of 4 (id: MI bits, dP, pAR v pNO, OR), top by MI:")
    for i,r in enumerate(RL):
        ro=np.where((nreg==1)&S[i])[0]; ro=ro[np.argsort(-MI[r][ro])]
        print(f"  {r} ({len(ro)}): "+", ".join(f"{int(x)}({MI[r][int(x)]:.3f}b,dP{dPo[r][int(x)]:.2f},{pAR[r][int(x)]:.2f}v{pNO[r][int(x)]:.2f},OR{ORr[r][int(x)]:.1f})" for x in ro[:12]))
    gl=np.where(nreg==4)[0]
    print(f"\n### GLOBAL latents — significant in all 4 ({len(gl)}): "+", ".join(str(int(x)) for x in gl[:20]))
    np.savez(OUT, mi=np.vstack([MI[r] for r in RL]), dP=np.vstack([dPo[r] for r in RL]),
             sig=S, regions=np.array(RL), nreg=nreg)
    print("saved", OUT)

if __name__=="__main__": main()
