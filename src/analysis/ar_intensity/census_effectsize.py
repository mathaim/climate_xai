"""Effect-size census (SAE via env): per region/concept, mean & SD of region-node-mean clamped
activation, split AR (qualifies==True, i.e. coverage>=0.5) vs non-AR (coverage_frac==0), one
consistent pass. Keeps ALL AR timesteps; subsamples non-AR-only steps (CAP_NONAR). 0<cov<0.5 is
skipped (neither). Checkpoints every 2000 steps -> resumes on restart. Output effectsize_{SAE}.npz."""
import os, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
OUT="results/ar_intensity/baseline"; SAE=os.environ.get("SAE","plain_L8")
CAP=int(os.environ.get("CAP_NONAR","12000")); CKPT=f"{OUT}/effectsize_{SAE}.ckpt.npz"
def main():
    os.makedirs(OUT,exist_ok=True); dev="cuda" if torch.cuda.is_available() else "cpu"
    print("SAE",SAE,"device",dev,"cap",CAP,flush=True)
    q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); setup=region_node_setup()
    arts={r:set(int(x) for x in q[(q.region==r)&(q.qualifies.astype(bool))].time_index) for r in REGIONS}
    nots={r:set(int(x) for x in q[(q.region==r)&(q.coverage_frac==0)].time_index) for r in REGIONS}
    ar_union=set().union(*arts.values()); no_union=set().union(*nots.values())
    nonar_only=sorted(no_union-ar_union)
    if len(nonar_only)>CAP: nonar_only=nonar_only[::max(1,len(nonar_only)//CAP)]
    proc=sorted(ar_union.union(nonar_only))
    print(f"proc {len(proc)} (AR_union {len(ar_union)}, nonAR_sample {len(nonar_only)})",flush=True)
    m,c,fmin,frng=load_sae(SAE,dev)
    rn={r:torch.as_tensor(np.asarray(setup[r]["nodes"]),dtype=torch.long,device=dev) for r in REGIONS}
    S={g:{r:{"s":np.zeros(4096),"sq":np.zeros(4096),"n":0} for r in REGIONS} for g in ("ar","no")}
    start=0
    if os.path.exists(CKPT):
        z=np.load(CKPT); start=int(z["j"][0])
        for g in ("ar","no"):
            for r in REGIONS:
                S[g][r]["s"]=z[f"{g}_{r}_s"].copy(); S[g][r]["sq"]=z[f"{g}_{r}_sq"].copy(); S[g][r]["n"]=int(z[f"{g}_{r}_n"][0])
        print("RESUMED at",start,flush=True)
    def save_ckpt(j):
        ck={"j":np.array([j])}
        for g in ("ar","no"):
            for r in REGIONS:
                ck[f"{g}_{r}_s"]=S[g][r]["s"]; ck[f"{g}_{r}_sq"]=S[g][r]["sq"]; ck[f"{g}_{r}_n"]=np.array([S[g][r]["n"]])
        np.savez(CKPT,**ck)
    for j in range(start,len(proc)):
        ti=proc[j]
        try: a=np.load(act_file(c,index_to_datetime(int(ti))),mmap_mode="r")
        except Exception: continue
        xr=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1); del a
        if fmin is not None: xr=(2.0*(xr-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad(): acts=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).clamp_min(0)
        for r in REGIONS:
            g="ar" if ti in arts[r] else ("no" if ti in nots[r] else None)
            if g is None: continue
            v=acts[rn[r]].mean(0).cpu().numpy(); S[g][r]["s"]+=v; S[g][r]["sq"]+=v*v; S[g][r]["n"]+=1
        del acts
        if (j+1)%2000==0: save_ckpt(j+1); print(f"{j+1}/{len(proc)} ckpt",flush=True)
    def stat(g,r):
        n=max(S[g][r]["n"],1); mu=S[g][r]["s"]/n; return mu, np.sqrt(np.maximum(S[g][r]["sq"]/n-mu*mu,0))
    save={}
    for r in REGIONS:
        mar,_=stat("ar",r); mno,sno=stat("no",r)
        save[f"{r}_mean_ar"]=mar.astype(np.float32); save[f"{r}_mean_no"]=mno.astype(np.float32)
        save[f"{r}_sd_no"]=sno.astype(np.float32)
        save[f"{r}_n_ar"]=np.array([S['ar'][r]['n']]); save[f"{r}_n_no"]=np.array([S['no'][r]['n']])
    np.savez(f"{OUT}/effectsize_{SAE}.npz",**save)
    if os.path.exists(CKPT): os.remove(CKPT)
    print("DONE",{r:(S['ar'][r]['n'],S['no'][r]['n']) for r in REGIONS},flush=True)
if __name__=="__main__": main()
