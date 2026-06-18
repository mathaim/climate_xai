"""Top local concept activation vs ground-truth IVT + AR mask, at each region's strongest AR."""
import os, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR, NPZ
import cartopy.crs as ccrs, cartopy.feature as cfeature
PIPE="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
OUT="results/ar_intensity/baseline"; PLOTS="results/ar_intensity/plots"; SAE="plain_L8"
conv=lambda x: x-360 if x>180 else x; PC=ccrs.PlateCarree()
CMAP=LinearSegmentedColormap.from_list("moist",["#e8f6f9","#9fdcc4","#4eb3d3","#2b6cb0","#3b3b98","#7d3c98","#c0392b"])
def geo(fig,pos):
    A=fig.add_subplot(1,3,pos,projection=PC)
    A.add_feature(cfeature.LAND,facecolor="#f0ede6",zorder=0); A.add_feature(cfeature.OCEAN,facecolor="#dce7ef",zorder=0)
    A.coastlines("110m",color="#555",lw=.6,zorder=4); return A
def main():
    os.makedirs(PLOTS,exist_ok=True); dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index(); d=np.load(NPZ)
    top=pd.read_csv(f"{OUT}/top_local_concepts.csv"); binned=pd.read_parquet(f"{PIPE}/ar_intensity_binned.parquet")
    m,c,fmin,frng=load_sae(SAE,dev); grid_lon=np.arange(0,360,0.25)
    for r in REGIONS:
        cc=int(top[top.region==r].sort_values("mag_ar",ascending=False).concept.iloc[0])
        sub=binned[binned.region==r]; ti=int(sub.loc[sub.max_ivt.idxmax(),"time_index"]); dt=index_to_datetime(ti)
        arr=np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        nlat=arr[:,lat_i]; nlon=np.array([conv(x) for x in arr[:,lon_i]]); iv=node_ivt(arr,qi,ui,vi,levels)
        act=np.load(act_file(c,dt),mmap_mode="r"); xr=np.ascontiguousarray(act).astype(np.float32).reshape(act.shape[0],-1)
        with torch.no_grad(): val=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()[:,cc]
        cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]; xs=[conv(x) for seg in lons for x in seg]
        ext=[min(xs)-4,max(xs)+4,la[0]-3,la[1]+3]
        mlat=d[f"{r}__lat"]; mlon=np.array([conv(x) for x in np.concatenate([grid_lon[(grid_lon>=x)&(grid_lon<=y)] for x,y in lons])])
        o=np.argsort(mlon); mlon=mlon[o]; mask=d[f"{r}__mask"][ti-1].astype(float)[:,o]; MLON,MLAT=np.meshgrid(mlon,mlat)
        ie=(nlon>=ext[0])&(nlon<=ext[1])&(nlat>=ext[2])&(nlat<=ext[3])
        vg=float(np.percentile(val[val>0],99)) if (val>0).any() else 1.0
        vz=float(np.percentile(val[ie&(val>0)],99)) if (ie&(val>0)).any() else vg
        vi_=float(np.percentile(iv[ie],98)) if ie.any() else 600.0
        fig=plt.figure(figsize=(18,5.2))
        A=geo(fig,1); A.set_global(); A.set_title(f"concept {cc} — global")
        A.tricontourf(nlon,nlat,np.clip(val,0,vg),levels=np.linspace(vg*.05,vg,12),cmap=CMAP,extend="max",transform=PC,zorder=2)
        A=geo(fig,2); A.set_extent(ext,crs=PC); A.set_title(f"concept {cc} — zoom")
        mp=A.tricontourf(nlon,nlat,np.clip(val,0,vz),levels=np.linspace(vz*.05,vz,12),cmap=CMAP,extend="max",transform=PC,zorder=2)
        A.contour(MLON,MLAT,mask,[.5],colors="#111",linewidths=1.6,transform=PC,zorder=6); A.gridlines(draw_labels=True,lw=.3,color="#bbb")
        fig.colorbar(mp,ax=A,shrink=.65,label="activation")
        A=geo(fig,3); A.set_extent(ext,crs=PC); A.set_title("IVT ground truth — zoom")
        ip=A.tricontourf(nlon,nlat,iv,levels=np.linspace(100,vi_,12),cmap="YlGnBu",extend="max",transform=PC,zorder=2)
        A.contour(MLON,MLAT,mask,[.5],colors="#111",linewidths=1.6,transform=PC,zorder=6); A.gridlines(draw_labels=True,lw=.3,color="#bbb")
        fig.colorbar(ip,ax=A,shrink=.65,label="IVT kg m$^{-1}$ s$^{-1}$")
        fig.suptitle(f"{r}: concept {cc} vs ground-truth AR (mask outline) — strongest AR {dt:%Y-%m-%d %Hz}",y=1.03)
        fig.savefig(f"{PLOTS}/truth_{r}_c{cc}.png",dpi=160,bbox_inches="tight"); plt.close(fig)
        print(f"{r}: concept {cc} {dt}",flush=True)
    print("TRUTH MAPS DONE")
if __name__=="__main__": main()
