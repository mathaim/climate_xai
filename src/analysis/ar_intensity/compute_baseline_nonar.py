"""Non-AR (coverage=0) region firing rates for L8 plain, all 3 measures -> baseline ratio denominator."""
import os, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
THR="results/ar_intensity/concept_thresholds"; OUT="results/ar_intensity/baseline"
SAE="plain_L8"; N_PER=3000; DEFS=["binary","magnitude","top10"]
def main():
    os.makedirs(OUT,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); setup=region_node_setup()
    rng=np.random.default_rng(0); samples={}
    for r in REGIONS:
        z=q[(q.region==r)&(q.coverage_frac==0)].time_index.to_numpy()
        samples[r]=set(int(x) for x in z)
    T=sorted(set().union(*samples.values())); print("non-AR timesteps:",len(T),flush=True)
    m,c,fmin,frng=load_sae(SAE,dev)
    p90=torch.from_numpy(np.load(f"{THR}/{SAE}_p90.npy")).to(dev)[None,:]
    rnodes={r:torch.as_tensor(np.asarray(setup[r]["nodes"]),dtype=torch.long,device=dev) for r in REGIONS}
    acc={r:{d:np.zeros(4096,np.float64) for d in DEFS} for r in REGIONS}; cnt={r:0 for r in REGIONS}
    for i,ti in enumerate(T):
        try: a=np.load(act_file(c,index_to_datetime(int(ti))),mmap_mode="r")
        except Exception: continue
        xr=np.ascontiguousarray(a).astype(np.float32); xr=xr.reshape(xr.shape[0],-1); del a
        with torch.no_grad():
            acts=encode(m,c["arch"],torch.from_numpy(xr).to(dev))
            ge0=(acts>0).float(); cl=acts.clamp_min(0); gt=(acts>p90).float()
            for r in REGIONS:
                if ti in samples[r]:
                    idx=rnodes[r]
                    acc[r]["binary"]+=ge0[idx].sum(0).cpu().numpy()
                    acc[r]["magnitude"]+=cl[idx].sum(0).cpu().numpy()
                    acc[r]["top10"]+=gt[idx].sum(0).cpu().numpy()
                    cnt[r]+=1
        del acts,ge0,cl,gt
        if (i+1)%1000==0: print(f"{i+1}/{len(T)}",flush=True)
    save={}
    for r in REGIONS:
        for d in DEFS: save[f"{r}_{d}_no"]=(acc[r][d]/max(cnt[r],1)).astype(np.float32)
        save[f"{r}_n_no"]=np.array([cnt[r]])
    np.savez(f"{OUT}/nonar_rates_{SAE}.npz", **save)
    print("NONAR DONE", {r:cnt[r] for r in REGIONS}, flush=True)
if __name__=="__main__": main()
