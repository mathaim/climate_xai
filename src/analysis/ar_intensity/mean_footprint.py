"""Mean footprint: featured concept mean activation + mean IVT + AR frequency over each region's >=50% ARs."""
import os, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR, NPZ
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
OUT="results/ar_intensity/baseline"; SAE="plain_L8"; N=40962
CONCEPT={"W_N_America":3069,"W_Europe":1008,"W_S_America":1115,"E_Australia":975}
def main():
    os.makedirs(OUT,exist_ok=True); dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index(); d=np.load(NPZ); masks={r:d[f"{r}__mask"] for r in REGIONS}
    qual=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); qual=qual[qual.qualifies]
    m,c,fmin,frng=load_sae(SAE,dev)
    acc={r:np.zeros(N) for r in REGIONS}; ivacc={r:np.zeros(N) for r in REGIONS}
    freq={r:np.zeros(masks[r].shape[1:]) for r in REGIONS}; cnt={r:0 for r in REGIONS}; miss=0; arr=None
    for i,(ti,grp) in enumerate(qual.groupby("time_index")):
        dt=index_to_datetime(int(ti)); rs=list(grp.region)
        try: arr=np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except Exception: miss+=1; continue
        iv=node_ivt(arr,qi,ui,vi,levels)
        a=np.load(act_file(c,dt),mmap_mode="r"); xr=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        with torch.no_grad(): acts=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()
        for r in rs:
            acc[r]+=acts[:,CONCEPT[r]]; ivacc[r]+=iv; freq[r]+=masks[r][int(ti)-1]; cnt[r]+=1
        if (i+1)%2000==0: print(f"{i+1} timesteps",flush=True)
    save={"nlat":arr[:,lat_i].astype(np.float32),"nlon":arr[:,lon_i].astype(np.float32)}
    for r in REGIONS:
        save[f"{r}_act"]=(acc[r]/max(cnt[r],1)).astype(np.float32)
        save[f"{r}_ivt"]=(ivacc[r]/max(cnt[r],1)).astype(np.float32)
        save[f"{r}_freq"]=(freq[r]/max(cnt[r],1)).astype(np.float32)
        save[f"{r}_n"]=np.array([cnt[r]]); save[f"{r}_concept"]=np.array([CONCEPT[r]])
    np.savez(f"{OUT}/mean_footprint_{SAE}.npz", **save)
    print("FOOTPRINT DONE",{r:cnt[r] for r in REGIONS},"miss",miss,flush=True)
if __name__=="__main__": main()
