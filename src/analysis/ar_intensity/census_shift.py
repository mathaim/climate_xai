"""
Node-level AR census with a CIRCULAR TIME-SHIFT null (accounts for temporal autocorrelation).
Reads pertimestep_{SAE}/{region}.npz. Per latent: MI(active; AR) over all (timestep,node). NULL: roll
the AR mask in time by NSHIFT random offsets (each >= MINSHIFT steps, i.e. beyond the decorrelation
time); family-wise ceiling = 99th pct of max-over-4096 null MI. AR-responsive = MI>ceiling AND excited
(p_AR>p_no). Prints per-region diagnostics, the census, and the LaTeX table. Saves census_shift_{SAE}.npz.
"""
import os, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
SAE=os.environ.get("SAE","plain_L8"); DIR=f"/scratch/euh7ys/climate_xai/concept_ivt/pertimestep_{SAE}"
NSHIFT=int(os.environ.get("NSHIFT","200")); MINSHIFT=int(os.environ.get("MINSHIFT","40"))
OUT=f"/scratch/euh7ys/climate_xai/concept_ivt/census_shift_{SAE}.npz"
NAME={"W_N_America":"Western North America","W_Europe":"Western Europe","W_S_America":"Western South America","E_Australia":"Eastern Australia"}
LAB={"plain_L8":("Standard","tab:standardcensus"),"matry_L8":("Matryoshka","tab:matrycensus")}

def mi_from(n11,Ac,R,N):
    n10=Ac-n11; n01=R-n11; n00=N-Ac-R+n11
    cells=[n11,n01,n10,n00]; cols=[Ac,N-Ac,Ac,N-Ac]; rows=[R,R,N-R,N-R]
    mi=np.zeros(np.shape(n11))
    for cnt,rw,cl in zip(cells,rows,cols):
        exp=(np.asarray(rw)*np.asarray(cl))/N
        with np.errstate(divide="ignore",invalid="ignore"): t=cnt*np.log2(np.where(cnt>0,cnt/exp,1.0))
        mi=mi+np.where(cnt>0,t,0.0)
    return mi/N

def main():
    RL=list(REGIONS); MI={}; DP={}; SIG={}; rng=np.random.default_rng(0)
    for r in RL:
        d=np.load(f"{DIR}/{r}.npz"); Apack=d["A"]; M=d["M"].astype(np.float32)
        T,n=M.shape; N=T*n; R=float(M.sum()); Mflat=M.reshape(N)
        p=R/N; H=-(p*np.log2(p)+(1-p)*np.log2(1-p)) if 0<p<1 else 0.0
        shifts=rng.integers(MINSHIFT,T-MINSHIFT,size=NSHIFT)
        Mmat=np.stack([np.roll(M,int(s),axis=0).reshape(N) for s in shifts])
        mi=np.zeros(4096); n11o=np.zeros(4096); Ac=np.zeros(4096); null=np.zeros((NSHIFT,4096))
        for c0 in range(0,4096,512):
            bits=np.unpackbits(Apack[:,:,c0//8:(c0+512)//8],axis=-1).astype(np.float32).reshape(N,512)
            Ac[c0:c0+512]=bits.sum(0); o=Mflat@bits; n11o[c0:c0+512]=o
            mi[c0:c0+512]=mi_from(o,Ac[c0:c0+512],R,N)
            null[:,c0:c0+512]=mi_from(Mmat@bits,Ac[c0:c0+512],R,N)
        ceil=float(np.percentile(null.max(1),99))
        p_ar=n11o/R; p_no=(Ac-n11o)/(N-R); dP=p_ar-p_no; sig=(mi>ceil)&(dP>0)
        MI[r]=mi; DP[r]=dP; SIG[r]=sig
        print(f"{r}: T={T} ARbase={p:.3f} H(AR)={H:.3f}b maxMI={mi.max():.4f}b ceil99={ceil:.4f}b #sig={int(sig.sum())}",flush=True)
    S=np.vstack([SIG[r] for r in RL]); nreg=S.sum(0); cnt={k:int((nreg==k).sum()) for k in (1,2,3,4)}
    print(f"\n### CENSUS ({SAE}, shift-null): active={int((nreg>=1).sum())} regional={cnt[1]} bi={cnt[2]} tri={cnt[3]} global={cnt[4]}")
    np.savez(OUT,mi=np.vstack([MI[r] for r in RL]),dP=np.vstack([DP[r] for r in RL]),sig=S,nreg=nreg,regions=np.array(RL))
    arch,lab=LAB.get(SAE,(SAE,"tab:census"))
    print("\n%----- LaTeX -----")
    print("\\begin{table}\n \\centering\n \\begin{tabular}{l c c c c c}\n \\toprule")
    print("  Region & Regional & Bi-regional & Tri-regional & Global & \\bf{Total} \\\\\n \\midrule")
    for i,r in enumerate(RL):
        c={k:int(((nreg==k)&S[i]).sum()) for k in (1,2,3,4)}
        print(f"   {NAME[r]} & {c[1]} & {c[2]} & {c[3]} & {c[4]} & \\bf{{{sum(c.values())}}}\\\\")
    print(f" \\midrule\n   Unique latents & {cnt[1]} & {cnt[2]} & {cnt[3]} & {cnt[4]} & \\bf{{{sum(cnt.values())}}} \\\\")
    print(f" \\bottomrule\n \\end{{tabular}}\n \\caption{{AR latent census ({arch} SAE, layer 8; circular time-shift null).}}\n \\label{{{lab}}}\n\\end{{table}}")
if __name__=="__main__": main()
