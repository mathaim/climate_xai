"""Stage 5 encode: per-SAE concept vectors for every qualifying (timestep, region)."""
import os, gc, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file, SAES
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
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
        if all(os.path.exists(f"{OUTDIR}/{n}_features.npy") for n in names):
            print(f"layer{layer}: already done, skipping",flush=True); continue
        models={n:load_sae(n,dev) for n in names}
        feats={n:[] for n in names}; meta=[]; nd=0
        c0=models[names[0]][1]
        for ti,grp in qual.groupby("time_index"):
            dt=index_to_datetime(int(ti))
            try: a=np.load(act_file(c0,dt), mmap_mode="r")
            except Exception: continue
            xr=np.ascontiguousarray(a[unodes]).astype(np.float32); del a; rs=list(grp.region)
            for n in names:
                m,c,fmin,frng=models[n]
                x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
                with torch.no_grad():
                    fired=(encode(m,c["arch"],torch.from_numpy(x).to(dev))>0).float().cpu().numpy()
                for r in rs: feats[n].append(fired[rmap[r]].mean(0))
                del fired, x
            for r in rs: meta.append((int(ti),r))
            del xr
            nd+=1
            if nd%2000==0: print(f"layer{layer}: {nd} timesteps",flush=True); gc.collect()
        md=pd.DataFrame(meta,columns=["time_index","region"]).merge(binned,on=["time_index","region"],how="left")
        for n in names:
            np.save(f"{OUTDIR}/{n}_features.npy", np.array(feats[n],dtype=np.float32))
            md.to_parquet(f"{OUTDIR}/{n}_meta.parquet")
            print(f"{n}: {len(feats[n])} rows saved",flush=True)
        del models, feats, meta, md; gc.collect()
    print("ALL DONE",flush=True)
if __name__=="__main__": main()
