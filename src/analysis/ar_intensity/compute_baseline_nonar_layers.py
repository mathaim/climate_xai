"""Non-AR region firing rates per SAE (region-node reads, ALL non-AR timesteps). Applies matry norm."""
import os, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
THR="results/ar_intensity/concept_thresholds"; OUT="results/ar_intensity/baseline"
SAES_TO_DO=os.environ.get("NONAR_SAES","plain_L0,plain_L15").split(","); DEFS=["binary","magnitude","top10"]
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); setup=region_node_setup()
    unodes=np.unique(np.concatenate([setup[r]["nodes"] for r in REGIONS])); pos={int(x):i for i,x in enumerate(unodes)}
    rmap={r:np.array([pos[int(x)] for x in setup[r]["nodes"]]) for r in REGIONS}
    noar={r:set(int(x) for x in q[(q.region==r)&(q.coverage_frac==0)].time_index) for r in REGIONS}
    keep=sorted(set().union(*noar.values())); print("non-AR timesteps (all):",len(keep),flush=True)
    for SAE in SAES_TO_DO:
        if os.path.exists(f"{OUT}/nonar_rates_{SAE}.npz"): print(SAE,"exists, skip",flush=True); continue
        m,c,fmin,frng=load_sae(SAE,dev); p90=torch.from_numpy(np.load(f"{THR}/{SAE}_p90.npy")).to(dev)[None,:]
        rt={r:torch.as_tensor(rmap[r],dtype=torch.long,device=dev) for r in REGIONS}
        acc={r:{d:np.zeros(4096) for d in DEFS} for r in REGIONS}; cnt={r:0 for r in REGIONS}
        for i,ti in enumerate(keep):
            try: a=np.load(act_file(c,index_to_datetime(int(ti))),mmap_mode="r")
            except Exception: continue
            xr=np.ascontiguousarray(a[unodes]).astype(np.float32).reshape(len(unodes),-1)
            x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
            with torch.no_grad():
                acts=encode(m,c["arch"],torch.from_numpy(x).to(dev)); ge0=(acts>0).float(); cl=acts.clamp_min(0); gt=(acts>p90).float()
                for r in REGIONS:
                    if ti in noar[r]:
                        idx=rt[r]
                        acc[r]["binary"]+=ge0[idx].sum(0).cpu().numpy(); acc[r]["magnitude"]+=cl[idx].sum(0).cpu().numpy(); acc[r]["top10"]+=gt[idx].sum(0).cpu().numpy(); cnt[r]+=1
            del acts,ge0,cl,gt
            if (i+1)%2000==0: print(f"{SAE} {i+1}/{len(keep)}",flush=True)
        save={}
        for r in REGIONS:
            for d in DEFS: save[f"{r}_{d}_no"]=(acc[r][d]/max(cnt[r],1)).astype(np.float32)
            save[f"{r}_n_no"]=np.array([cnt[r]])
        np.savez(f"{OUT}/nonar_rates_{SAE}.npz",**save); print(SAE,"DONE",{r:cnt[r] for r in REGIONS},flush=True)
    print("ALL NONAR DONE")
if __name__=="__main__": main()
