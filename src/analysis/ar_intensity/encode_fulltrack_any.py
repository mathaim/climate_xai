"""Full-record region encode (max+mean pooled) for ANY SAE name (arg). track_<SAE>_<region>.npz"""
import os, sys, numpy as np, pandas as pd, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup
OUT="/scratch/euh7ys/climate_xai/concept_ivt"; TOTAL=56700
IVT_PARQUET="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/ar_intensity_full.parquet"
def main():
    SAE=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("SAE",SAE,"device",dev,flush=True)
    setup=region_node_setup(); m,c,fmin,frng=load_sae(SAE,dev)
    iv=pd.read_parquet(IVT_PARQUET)
    mx={r:dict(zip(g.time_index,g.max_ivt)) for r,g in iv.groupby("region")}
    mn={r:dict(zip(g.time_index,g.mean_ivt)) for r,g in iv.groupby("region")}
    AMAX={r:np.zeros((TOTAL,4096),np.float32) for r in REGIONS}; AMEAN={r:np.zeros((TOTAL,4096),np.float32) for r in REGIONS}
    IVT={r:np.full(TOTAL,np.nan,np.float32) for r in REGIONS}; IVTM={r:np.full(TOTAL,np.nan,np.float32) for r in REGIONS}
    month=np.zeros(TOTAL,np.int16); tindex=np.arange(1,TOTAL+1); miss=0
    for n,i in enumerate(range(1,TOTAL+1)):
        dt=index_to_datetime(i); month[n]=dt.month
        try: a=np.load(act_file(c,dt),mmap_mode="r")
        except Exception: miss+=1; continue
        for r in REGIONS:
            nodes=setup[r]["nodes"]; xr=np.ascontiguousarray(a[nodes]).astype(np.float32).reshape(len(nodes),-1)
            x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
            with torch.no_grad(): acts=encode(m,c["arch"],torch.from_numpy(x).to(dev)).cpu().numpy()
            AMAX[r][n]=acts.max(0); AMEAN[r][n]=acts.mean(0)
            IVT[r][n]=mx[r].get(int(i),np.nan); IVTM[r][n]=mn[r].get(int(i),np.nan)
        if n%2000==0: print(n,"/",TOTAL,"miss",miss,flush=True)
    for r in REGIONS:
        np.savez(f"{OUT}/track_{SAE}_{r}.npz",A_max=AMAX[r],A_mean=AMEAN[r],ivt=IVT[r],ivt_mean=IVTM[r],tindex=tindex,month=month)
        print("saved",r,flush=True)
    print("ENCODE DONE missing",miss)
if __name__=="__main__": main()
