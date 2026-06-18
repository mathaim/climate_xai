"""Snapshot: each region's top LOCAL concept activation at its strongest AR (smooth, mapped)."""
import os, glob, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
OUT="results/ar_intensity/baseline"; PLOTS="results/ar_intensity/plots"; SAE="plain_L8"
conv=lambda x: x-360 if x>180 else x
CMAP=LinearSegmentedColormap.from_list("moist",["#e8f6f9","#9fdcc4","#4eb3d3","#2b6cb0","#3b3b98","#7d3c98","#c0392b"])
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature; HAVE=True
except Exception: HAVE=False
def axis(fig,pos):
    if HAVE:
        A=fig.add_subplot(1,2,pos,projection=ccrs.PlateCarree())
        A.add_feature(cfeature.LAND,facecolor="#f0ede6",zorder=0)
        A.add_feature(cfeature.OCEAN,facecolor="#dce7ef",zorder=0)
        try: A.coastlines("110m",color="#555",linewidth=.6,zorder=4)
        except Exception: pass
        return A, dict(transform=ccrs.PlateCarree())
    return fig.add_subplot(1,2,pos), {}
def main():
    os.makedirs(PLOTS,exist_ok=True); dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,"cartopy",HAVE,flush=True)
    idx,_,_,_,_,lat_i,lon_i=load_channel_index()
    era=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat=era[:,lat_i]; nlon=np.array([conv(x) for x in era[:,lon_i]])
    top=pd.read_csv(f"{OUT}/top_local_concepts.csv"); binned=pd.read_parquet(f"{PIPE}/ar_intensity_binned.parquet")
    m,c,fmin,frng=load_sae(SAE,dev)
    for r in REGIONS:
        cc=int(top[top.region==r].sort_values("mag_ar",ascending=False).concept.iloc[0])
        sub=binned[binned.region==r]; ti=int(sub.loc[sub.max_ivt.idxmax(),"time_index"]); dt=index_to_datetime(ti)
        a=np.load(act_file(c,dt),mmap_mode="r"); xr=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        with torch.no_grad():
            val=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()[:,cc]
        vmax=float(np.percentile(val[val>0],99)) if (val>0).any() else 1.0
        levels=np.linspace(vmax*0.05,vmax,12)
        cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]; xs=[conv(x) for seg in lons for x in seg]
        fig=plt.figure(figsize=(15,5.5))
        for pos,zoom in [(1,False),(2,True)]:
            A,kw=axis(fig,pos)
            mp=A.tricontourf(nlon,nlat,np.clip(val,0,vmax),levels=levels,cmap=CMAP,extend="max",zorder=2,**kw)
            for x0,x1 in lons:
                A.plot([conv(x0),conv(x1),conv(x1),conv(x0),conv(x0)],[la[0],la[0],la[1],la[1],la[0]],c="#111",lw=1.6,zorder=5,**kw)
            if zoom:
                ext=[min(xs)-4,max(xs)+4,la[0]-3,la[1]+3]
                if HAVE: A.set_extent(ext,crs=ccrs.PlateCarree()); A.gridlines(draw_labels=True,lw=.3,color="#999")
                else: A.set_xlim(ext[0],ext[1]); A.set_ylim(ext[2],ext[3])
                A.set_title("region zoom")
            else:
                if HAVE: A.set_global()
                else: A.set_xlim(-180,180); A.set_ylim(-90,90)
                A.set_title("global")
        fig.colorbar(mp,ax=fig.axes,shrink=.65,label=f"concept {cc} activation")
        fig.suptitle(f"{r}: top local concept {cc} at strongest AR ({dt:%Y-%m-%d %Hz})",y=.98)
        fig.savefig(f"{PLOTS}/snapshot_{r}_c{cc}.png",dpi=160,bbox_inches="tight"); plt.close(fig)
        print(f"{r}: concept {cc} {dt}",flush=True)
    print("SNAPSHOTS DONE")
if __name__=="__main__": main()
