"""Find core intensity parents active in >=3 regions (global) and list their outer
children grouped by the child's specialized region -> does one global parent fan out
into different-region children?"""
import numpy as np
from collections import Counter
COF="/scratch/euh7ys/climate_xai/cofire/cofire_matry_L8.npz"
CORR="/home/euh7ys/climate_xai/results/ar_intensity/corr"
REG=["W_N_America","W_Europe","W_S_America","E_Australia"]
def main():
    d=np.load(COF); C=d["cofire"].astype(float); f=d["fire"].astype(float); N=float(d["nodes"][0])
    R=np.vstack([np.abs(np.load(f"{CORR}/matry_L8_region_magnitude_{r}.npy")) for r in REG])
    core=np.arange(512); outer=np.arange(2048,4096); base=f/N
    nreg_core=(R[:,:512]>0.25).sum(0)
    gp=core[nreg_core>=3]
    print("global core parents (|r|>0.25 in >=3 regions):", list(gp))
    for p in gp:
        if f[p]<50: continue
        prof={REG[i][:4]:round(float(R[i,p]),2) for i in range(4)}
        Ppc=C[p,outer]/np.maximum(f[outer],1); kids=[]
        for c in outer[(Ppc>0.6)&(f[outer]>50)]:
            fwd=C[p,c]/f[c]; rev=C[p,c]/f[p]
            if rev>0.4*fwd or f[p]<=f[c]: continue
            cmax=R[:,c].max(); 
            if cmax<0.15: continue
            kids.append((c,fwd,REG[int(np.argmax(R[:,c]))][:4],round(float(cmax),2),int((R[:,c]>0.2).sum())))
        if not kids: continue
        print(f"\nGLOBAL PARENT {p}  fires {f[p]/N*100:.2f}%  |r|={prof}")
        print(f"  {len(kids)} children, regions covered: {dict(Counter(k[2] for k in kids))}")
        for c,fwd,cr,cmax,cn in sorted(kids,key=lambda x:-x[1])[:12]:
            print(f"   child {c}: P(p|c)={fwd:.2f} region={cr} child|r|={cmax} #reg={cn}")
    print("\nDONE")
if __name__=="__main__":
    main()
