"""Stage 5 encode: Binary/Magnitude/Top10 at REGION and GLOBAL node scope, per (timestep, region).
Full-mesh encode; heavy sums on-device. Set ONLY_LAYER=N to process one layer (for parallel jobs)."""
import os, gc, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file, SAES
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
THR="results/ar_intensity/concept_thresholds"
DEFS=["binary","magnitude","top10"]; SCOPES=["region","global"]; KEYS=[f"{s}_{d}" for s in SCOPES for d in DEFS]
def main():
    os.makedirs(OUTDIR,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    only=os.environ.get("ONLY_LAYER")
    qual=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); qual=qual[qual.qualifies]
    binned=pd.read_parquet(f"{PIPE}/ar_intensity_binned.parquet")[["time_index","region","intensity_bin","max_ivt"]]
    setup=region_node_setup()
    layers={}
    for name,c in SAES.items(): layers.setdefault(c["layer"],[]).append(name)
    for layer,names in layers.items():
        if only is not None and int(only)!=layer: continue
        if all(os.path.exists(f"{OUTDIR}/{n}_features_{k}.npy") for n in names for k in KEYS):
            print(f"layer{layer}: done, skip",flush=True); continue
        models={n:load_sae(n,dev) for n in names}
        p90={n:torch.from_numpy(np.load(f"{THR}/{n}_p90.npy")).to(dev)[None,:] for n in names}
        rnodes={r:torch.as_tensor(np.asarray(setup[r]["nodes"]),dtype=torch.long,device=dev) for r in REGIONS}
        feats={n:{k:[] for k in KEYS} for n in names}; meta=[]; nd=0
        c0=models[names[0]][1]
        for ti,grp in qual.groupby("time_index"):
            try: a=np.load(act_file(c0,index_to_datetime(int(ti))),mmap_mode="r")
            except Exception: continue
            xr=np.ascontiguousarray(a).astype(np.float32); xr=xr.reshape(xr.shape[0],-1); del a
            rs=list(grp.region)
            for n in names:
                m,c,fmin,frng=models[n]
                x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
                with torch.no_grad():
                    acts=encode(m,c["arch"],torch.from_numpy(x).to(dev))
                    th=p90[n]
                    ge0=(acts>0).float(); cl=acts.clamp_min(0); gt=(acts>th).float()
                    gbin=ge0.sum(0).cpu().numpy(); gmag=cl.sum(0).cpu().numpy(); gtop=gt.sum(0).cpu().numpy()
                    for r in rs:
                        idx=rnodes[r]
                        feats[n]["region_binary"].append(ge0[idx].sum(0).cpu().numpy())
                        feats[n]["region_magnitude"].append(cl[idx].sum(0).cpu().numpy())
                        feats[n]["region_top10"].append(gt[idx].sum(0).cpu().numpy())
                        feats[n]["global_binary"].append(gbin); feats[n]["global_magnitude"].append(gmag); feats[n]["global_top10"].append(gtop)
                del acts,ge0,cl,gt
            for r in rs: meta.append((int(ti),r))
            del xr; nd+=1
            if nd%2000==0: print(f"layer{layer}: {nd} timesteps",flush=True); gc.collect()
        md=pd.DataFrame(meta,columns=["time_index","region"]).merge(binned,on=["time_index","region"],how="left")
        for n in names:
            for k in KEYS: np.save(f"{OUTDIR}/{n}_features_{k}.npy", np.asarray(feats[n][k],dtype=np.float32))
            md.to_parquet(f"{OUTDIR}/{n}_meta.parquet")
            print(f"{n}: {len(feats[n]['region_binary'])} rows x{len(KEYS)} arrays",flush=True)
        del models,feats,meta,md,p90; gc.collect()
    print("ALL DONE",flush=True)
if __name__=="__main__": main()
