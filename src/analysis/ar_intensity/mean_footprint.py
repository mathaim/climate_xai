"""Mean footprint (fast): featured concept mean activation + AR frequency over each region's >=50% ARs."""
import os, glob, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR, NPZ
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
OUT="results/ar_intensity/baseline"; SAE="plain_L8"; N=40962; N_AR=800
CONCEPT={"W_N_America":3069,"W_Europe":1008,"W_S_America":1115,"E_Australia":975}
def main():
    os.makedirs(OUT,exist_ok=True); dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index(); d=np.load(NPZ); masks={r:d[f"{r}__mask"] for r in REGIONS}
    era0=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat=era0[:,lat_i]; nlon=era0[:,lon_i]
    qual=pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); qual=qual[qual.qualifies]
    rng=np.random.default_rng(0); samp={}
    for r in REGIONS:
        ts=qual[qual.region==r].time_index.to_numpy(); samp[r]=set(int(x) for x in rng.choice(ts,min(N_AR,len(ts)),replace=False))
    keep=sorted(set().union(*samp.values())); print("sampled timesteps:",len(keep),flush=True)
    m,c,fmin,frng=load_sae(SAE,dev)
    acc={r:np.zeros(N) for r in REGIONS}; freq={r:np.zeros(masks[r].shape[1:]) for r in REGIONS}; cnt={r:0 for r in REGIONS}
    for i,ti in enumerate(keep):
        dt=index_to_datetime(int(ti)); rs=[r for r in REGIONS if ti in samp[r]]
        try: a=np.load(act_file(c,dt),mmap_mode="r")
        except Exception: continue
        xr=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        with torch.no_grad(): acts=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()
        for r in rs: acc[r]+=acts[:,CONCEPT[r]]; freq[r]+=masks[r][int(ti)-1]; cnt[r]+=1
        if (i+1)%4000==0: print(f"{i+1} timesteps",flush=True)
    save={"nlat":nlat.astype(np.float32),"nlon":nlon.astype(np.float32)}
    for r in REGIONS:
        save[f"{r}_act"]=(acc[r]/max(cnt[r],1)).astype(np.float32)
        save[f"{r}_freq"]=(freq[r]/max(cnt[r],1)).astype(np.float32)
        save[f"{r}_n"]=np.array([cnt[r]]); save[f"{r}_concept"]=np.array([CONCEPT[r]])
    np.savez(f"{OUT}/mean_footprint_{SAE}.npz", **save)
    print("FOOTPRINT DONE",{r:cnt[r] for r in REGIONS},flush=True)
if __name__=="__main__": main()
