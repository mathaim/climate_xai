"""Stage 5 encode: per-SAE Binary/Magnitude/Top10 concept features per (timestep, region)."""
import os, gc, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file, SAES
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
THR="results/ar_intensity/concept_thresholds"; DEFS=["binary","magnitude","top10"]
def main():
    os.makedirs(OUTDIR,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    qual=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); qual=qual[qual.qualifies]
    binned=pd.read_parquet(f"{PIPE}/ar_intensity_binned.parquet")[["time_index","region","intensity_bin","max_ivt"]]
    setup=region_node_setup()
    unodes=np.unique(np.concatenate([setup[r]["nodes"] for r in REGIONS]))
    pos={int(n):i for i,n in enumerate(unodes)}
    rmap={r:np.array([pos[int(n)] for n in setup[r]["nodes"]]) for r in REGIONS}
    layers={}
    for name,c in SAES.items(): layers.setdefault(c["layer"],[]).append(name)
    for layer,names in layers.items():
        if all(os.path.exists(f"{OUTDIR}/{n}_features_{d}.npy") for n in names for d in DEFS):
            print(f"layer{layer}: already done, skipping",flush=True); continue
        models={n:load_sae(n,dev) for n in names}
        p90={n:np.load(f"{THR}/{n}_p90.npy")[None,:] for n in names}
        feats={n:{d:[] for d in DEFS} for n in names}; meta=[]; nd=0
        c0=models[names[0]][1]
        for ti,grp in qual.groupby("time_index"):
            dt=index_to_datetime(int(ti))
            try: a=np.load(act_file(c0,dt), mmap_mode="r")
            except Exception: continue
            xr=np.ascontiguousarray(a[unodes]).astype(np.float32); xr=xr.reshape(xr.shape[0],-1); del a
            rs=list(grp.region)
            for n in names:
                m,c,fmin,frng=models[n]
                x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
                with torch.no_grad():
                    acts=encode(m,c["arch"],torch.from_numpy(x).to(dev)).cpu().numpy()
                th=p90[n]
                for r in rs:
                    sub=acts[rmap[r]]
                    feats[n]["binary"].append((sub>0).sum(0).astype(np.float32))
                    feats[n]["magnitude"].append(sub.sum(0).astype(np.float32))
                    feats[n]["top10"].append((sub>th).sum(0).astype(np.float32))
                del acts, x
            for r in rs: meta.append((int(ti),r))
            del xr; nd+=1
            if nd%2000==0: print(f"layer{layer}: {nd} timesteps",flush=True); gc.collect()
        md=pd.DataFrame(meta,columns=["time_index","region"]).merge(binned,on=["time_index","region"],how="left")
        for n in names:
            for d in DEFS:
                np.save(f"{OUTDIR}/{n}_features_{d}.npy", np.array(feats[n][d],dtype=np.float32))
            md.to_parquet(f"{OUTDIR}/{n}_meta.parquet")
            print(f"{n}: {len(feats[n]['binary'])} rows x3 defs saved",flush=True)
        del models,feats,meta,md,p90; gc.collect()
    print("ALL DONE",flush=True)
if __name__=="__main__": main()
