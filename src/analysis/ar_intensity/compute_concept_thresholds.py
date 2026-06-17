"""Per-concept p90 over NONZERO activations, matching ar_analysis_measures.py
(subsample 1000/concept/file, ~50 files, percentile 90, p90=0 if never active)."""
import os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import SAES, load_sae, encode, act_file
from src.analysis.ar_intensity.regions import index_to_datetime
OUT="results/ar_intensity/concept_thresholds"; N_FILES=50; N_CONCEPTS=4096; CAP=1000
def main():
    os.makedirs(OUT,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    rng=np.random.default_rng(0)
    layers={}
    for name,c in SAES.items(): layers.setdefault(c["layer"],[]).append(name)
    for layer,names in layers.items():
        if all(os.path.exists(f"{OUT}/{n}_p90.npy") for n in names):
            print(f"layer{layer}: exists, skip",flush=True); continue
        models={n:load_sae(n,dev) for n in names}; c0=models[names[0]][1]
        allvals={n:[[] for _ in range(N_CONCEPTS)] for n in names}; used=0; ti=0
        while used<N_FILES and ti<60000:
            ti+=1
            try: a=np.load(act_file(c0,index_to_datetime(ti)),mmap_mode="r")
            except Exception: continue
            xr=np.ascontiguousarray(a).astype(np.float32); xr=xr.reshape(xr.shape[0],-1); del a
            for n in names:
                m,c,fmin,frng=models[n]
                x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
                with torch.no_grad():
                    acts=encode(m,c["arch"],torch.from_numpy(x).to(dev)).cpu().numpy()
                for cc in range(N_CONCEPTS):
                    col=acts[:,cc]; nz=col[col>0]
                    if len(nz):
                        if len(nz)>CAP: nz=nz[rng.choice(len(nz),CAP,replace=False)]
                        allvals[n][cc].append(nz)
            used+=1
            if used%10==0: print(f"layer{layer}: {used}/{N_FILES} files",flush=True)
        for n in names:
            p90=np.zeros(N_CONCEPTS,np.float32)
            for cc in range(N_CONCEPTS):
                if allvals[n][cc]: p90[cc]=np.percentile(np.concatenate(allvals[n][cc]),90)
            np.save(f"{OUT}/{n}_p90.npy",p90)
            print(f"{n}: p90 saved ({int((p90>0).sum())} alive, median={np.median(p90[p90>0]):.4f})",flush=True)
        del models,allvals
    print("THRESHOLDS DONE",flush=True)
if __name__=="__main__": main()
