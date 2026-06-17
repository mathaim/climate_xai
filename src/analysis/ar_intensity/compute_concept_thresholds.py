"""Per-concept 90th-percentile threshold over NONZERO activations (global, sampled), per SAE.
Used by the Top-10% firing definition: count(a > p90). Saves {sae}_p90.npy (4096,)."""
import os, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import SAES, load_sae, encode, act_file
from src.analysis.ar_intensity.regions import index_to_datetime
OUT="results/ar_intensity/concept_thresholds"; N_SAMPLE=80; TOTAL=56700
def main():
    os.makedirs(OUT,exist_ok=True); dev="cpu"
    cand=np.random.default_rng(0).choice(TOTAL, N_SAMPLE*3, replace=False)+1
    layers={}
    for name,c in SAES.items(): layers.setdefault(c["layer"],[]).append(name)
    for layer,names in layers.items():
        if all(os.path.exists(f"{OUT}/{n}_p90.npy") for n in names):
            print(f"layer{layer}: thresholds exist, skipping",flush=True); continue
        models={n:load_sae(n,dev) for n in names}; c0=models[names[0]][1]
        vals={n:[] for n in names}; cids={n:[] for n in names}; used=0
        for ti in cand:
            if used>=N_SAMPLE: break
            try: a=np.load(act_file(c0, index_to_datetime(int(ti))), mmap_mode="r")
            except Exception: continue
            xr=np.ascontiguousarray(a).astype(np.float32); xr=xr.reshape(xr.shape[0],-1); del a
            for n in names:
                m,c,fmin,frng=models[n]
                x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
                with torch.no_grad():
                    acts=encode(m,c["arch"],torch.from_numpy(x).to(dev)).cpu().numpy()
                ni,nc=np.nonzero(acts>0)
                vals[n].append(acts[ni,nc].astype(np.float32)); cids[n].append(nc.astype(np.int16))
            used+=1
            if used%20==0: print(f"layer{layer}: {used} timesteps",flush=True)
        for n in names:
            df=pd.DataFrame({"c":np.concatenate(cids[n]),"v":np.concatenate(vals[n])})
            p=df.groupby("c").v.quantile(0.90)
            thr=np.full(4096, np.inf, np.float32); thr[p.index.values]=p.values.astype(np.float32)
            np.save(f"{OUT}/{n}_p90.npy", thr)
            fin=thr[np.isfinite(thr)]
            print(f"{n}: p90 saved ({len(fin)} active concepts, median thr={np.median(fin):.4f})",flush=True)
        del models,vals,cids
    print("THRESHOLDS DONE",flush=True)
if __name__=="__main__": main()
