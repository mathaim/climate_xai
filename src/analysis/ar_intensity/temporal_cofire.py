"""Temporal (per-input) co-firing for the two exemplar families:
P(parent fires at any node | child fires at any node) per timestep, over a sample of the record.
Full-mesh matry_L8 encode. Fills the 'per input' column of tab:matryoshkaconcepts (expected ~1)."""
import os, glob, datetime as DT, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
SAE="matry_L8"; N=int(os.environ.get("N","8000"))
FAM={340:[3481,3948,3675], 99:[1454,3392,2722]}
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("dev",dev,flush=True)
    m,c,fmin,frng=load_sae(SAE,dev)
    files=sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
    sel=[files[i] for i in np.linspace(0,len(files)-1,min(N,len(files))).astype(int)]
    print(f"encoding {len(sel)} timesteps (full mesh)",flush=True)
    fires=np.zeros((len(sel),4096),bool)
    for k,f in enumerate(sel):
        a=np.load(f,mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        if fmin is not None: x=(2.0*(x-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad(): acts=encode(m,c["arch"],torch.from_numpy(x).to(dev))
        fires[k]=(acts>0).any(0).cpu().numpy()
        if (k+1)%1000==0: print(f"  {k+1}/{len(sel)}",flush=True)
    print(f"\nTemporal (per-input) co-firing over {len(sel)} timesteps  [P(par fires | child fires)]:")
    for p,cs in FAM.items():
        fp=fires[:,p]
        print(f"  parent {p}: fires on {int(fp.sum())}/{len(sel)} timesteps")
        for ch in cs:
            fc=fires[:,ch]; pcf=(fp&fc).sum()/max(fc.sum(),1)
            print(f"    {p} <- {ch}: {pcf:.3f}   (child fires {int(fc.sum())}/{len(sel)})")
if __name__=="__main__": main()
