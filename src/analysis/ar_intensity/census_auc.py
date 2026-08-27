"""AUC job (SAE via env): per concept x region, histogram of region-node-mean clamped activation over
AR (qualifies) vs non-AR (coverage==0) timesteps -> AUC=P(v_AR>v_nonAR), no cutoff, heavy-tail robust.
Also F = concept overall mean activation (firing strength). Checkpointed. Output auc_{SAE}.npz."""
import os, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
OUT="results/ar_intensity/baseline"; SAE=os.environ.get("SAE","plain_L8")
CAP=int(os.environ.get("CAP_NONAR","12000"))
LOG_EDGES=np.logspace(-6,1.0,400); NBIN=len(LOG_EDGES)+1; CKPT=f"{OUT}/auc_{SAE}.ckpt.npz"
def main():
    os.makedirs(OUT,exist_ok=True); dev="cuda" if torch.cuda.is_available() else "cpu"
    print("SAE",SAE,"dev",dev,"nbin",NBIN,flush=True)
    q=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); setup=region_node_setup()
    arts={r:set(int(x) for x in q[(q.region==r)&(q.qualifies.astype(bool))].time_index) for r in REGIONS}
    nots={r:set(int(x) for x in q[(q.region==r)&(q.coverage_frac==0)].time_index) for r in REGIONS}
    ar_union=set().union(*arts.values()); no_union=set().union(*nots.values())
    nonar_only=sorted(no_union-ar_union)
    if len(nonar_only)>CAP: nonar_only=nonar_only[::max(1,len(nonar_only)//CAP)]
    proc=sorted(ar_union.union(nonar_only)); print("proc",len(proc),flush=True)
    m,c,fmin,frng=load_sae(SAE,dev)
    rn={r:torch.as_tensor(np.asarray(setup[r]["nodes"]),dtype=torch.long,device=dev) for r in REGIONS}
    H={g:{r:np.zeros((4096,NBIN),np.int64) for r in REGIONS} for g in ("ar","no")}
    Fsum=np.zeros(4096,np.float64); Fn=np.array([0]); ar=np.arange(4096); start=0
    if os.path.exists(CKPT):
        z=np.load(CKPT); start=int(z["j"][0]); Fsum=z["Fsum"].copy(); Fn=z["Fn"].copy()
        for g in ("ar","no"):
            for r in REGIONS: H[g][r]=z[f"{g}_{r}"].copy()
        print("RESUMED",start,flush=True)
    def save_ckpt(j):
        ck={"j":np.array([j]),"Fsum":Fsum,"Fn":Fn}
        for g in ("ar","no"):
            for r in REGIONS: ck[f"{g}_{r}"]=H[g][r]
        np.savez(CKPT,**ck)
    for j in range(start,len(proc)):
        ti=proc[j]
        try: a=np.load(act_file(c,index_to_datetime(int(ti))),mmap_mode="r")
        except Exception: continue
        xr=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1); del a
        if fmin is not None: xr=(2.0*(xr-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad(): acts=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).clamp_min(0)
        Fsum+=acts.mean(0).cpu().numpy(); Fn[0]+=1
        for r in REGIONS:
            g="ar" if ti in arts[r] else ("no" if ti in nots[r] else None)
            if g is None: continue
            v=acts[rn[r]].mean(0).cpu().numpy(); H[g][r][ar,np.digitize(v,LOG_EDGES)]+=1
        del acts
        if (j+1)%3000==0: save_ckpt(j+1); print(f"{j+1}/{len(proc)} ckpt",flush=True)
    save={"Fsum":Fsum,"Fn":Fn}
    for g in ("ar","no"):
        for r in REGIONS: save[f"{g}_{r}"]=H[g][r].astype(np.int32)
    np.savez(f"{OUT}/auc_{SAE}.npz",**save)
    if os.path.exists(CKPT): os.remove(CKPT)
    print("DONE Ns:",{r:(int(H['ar'][r].sum(1).max()),int(H['no'][r].sum(1).max())) for r in REGIONS},flush=True)
if __name__=="__main__": main()
