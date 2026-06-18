"""Snapshot: each region's top LOCAL concept activation at its strongest AR timestep."""
import os, glob, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
OUT="results/ar_intensity/baseline"; PLOTS="results/ar_intensity/plots"; SAE="plain_L8"
conv=lambda x: x-360 if x>180 else x
def main():
    os.makedirs(PLOTS,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,_,_,_,_,lat_i,lon_i=load_channel_index()
    era=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat=era[:,lat_i]; nlon=np.array([conv(x) for x in era[:,lon_i]])
    top=pd.read_csv(f"{OUT}/top_local_concepts.csv")
    binned=pd.read_parquet(f"{PIPE}/ar_intensity_binned.parquet")
    m,c,fmin,frng=load_sae(SAE,dev)
    for r in REGIONS:
        cc=int(top[top.region==r].sort_values("mag_ar",ascending=False).concept.iloc[0])
        sub=binned[binned.region==r]; ti=int(sub.loc[sub.max_ivt.idxmax(),"time_index"]); dt=index_to_datetime(ti)
        a=np.load(act_file(c,dt),mmap_mode="r"); xr=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        with torch.no_grad():
            val=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()[:,cc]
        vmax=np.percentile(val[val>0],99) if (val>0).any() else 1.0
        cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]; xs=[conv(x) for seg in lons for x in seg]
        fig,ax=plt.subplots(1,2,figsize=(15,5))
        for k,zoom in enumerate([False,True]):
            A=ax[k]; sc=A.scatter(nlon,nlat,c=val,s=5,cmap="magma",vmin=0,vmax=vmax)
            for x0,x1 in lons:
                xx0,xx1=conv(x0),conv(x1)
                A.plot([xx0,xx1,xx1,xx0,xx0],[la[0],la[0],la[1],la[1],la[0]],c="#00ffff",lw=1.5)
            A.set_xlabel("longitude"); A.set_ylabel("latitude")
            if zoom: A.set_xlim(min(xs)-20,max(xs)+20); A.set_ylim(la[0]-15,la[1]+15); A.set_title("region zoom")
            else: A.set_xlim(-180,180); A.set_ylim(-90,90); A.set_title("global")
        fig.colorbar(sc,ax=ax,shrink=.7,label=f"concept {cc} activation")
        fig.suptitle(f"{r}: top local concept {cc} at strongest AR ({dt:%Y-%m-%d %Hz})")
        fig.savefig(f"{PLOTS}/snapshot_{r}_c{cc}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        print(f"{r}: concept {cc}, strongest AR {dt}",flush=True)
    print("SNAPSHOTS DONE")
if __name__=="__main__": main()
