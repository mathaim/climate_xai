"""Mean-footprint figures: featured concept mean activation (global+zoom) + mean IVT, AR>50% freq contour."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.ivt_pipeline import NPZ
import cartopy.crs as ccrs, cartopy.feature as cfeature
OUT="results/ar_intensity/baseline"; PLOTS="results/ar_intensity/plots"; SAE="plain_L8"
conv=lambda x: x-360 if x>180 else x; PC=ccrs.PlateCarree()
CMAP=LinearSegmentedColormap.from_list("moist",["#e8f6f9","#9fdcc4","#4eb3d3","#2b6cb0","#3b3b98","#7d3c98","#c0392b"])
def geo(fig,pos):
    A=fig.add_subplot(1,3,pos,projection=PC); A.add_feature(cfeature.LAND,facecolor="#f0ede6",zorder=0)
    A.add_feature(cfeature.OCEAN,facecolor="#dce7ef",zorder=0); A.coastlines("110m",color="#555",lw=.6,zorder=4); return A
def main():
    z=np.load(f"{OUT}/mean_footprint_{SAE}.npz"); d=np.load(NPZ); grid_lon=np.arange(0,360,0.25)
    nlat=z["nlat"]; nlon=np.array([conv(x) for x in z["nlon"]])
    for r in REGIONS:
        cc=int(z[f"{r}_concept"][0]); val=z[f"{r}_act"]; iv=z[f"{r}_ivt"]; freq=z[f"{r}_freq"]; n=int(z[f"{r}_n"][0])
        cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]; xs=[conv(x) for seg in lons for x in seg]
        ext=[min(xs)-4,max(xs)+4,la[0]-3,la[1]+3]
        mlat=d[f"{r}__lat"]; mlon=np.array([conv(x) for x in np.concatenate([grid_lon[(grid_lon>=x)&(grid_lon<=y)] for x,y in lons])])
        o=np.argsort(mlon); mlon=mlon[o]; freq=freq[:,o]; MLON,MLAT=np.meshgrid(mlon,mlat)
        ie=(nlon>=ext[0])&(nlon<=ext[1])&(nlat>=ext[2])&(nlat<=ext[3])
        vg=float(np.percentile(val[val>0],99)) if (val>0).any() else 1.0
        vz=float(np.percentile(val[ie&(val>0)],99)) if (ie&(val>0)).any() else vg
        vi_=float(np.percentile(iv[ie],98)) if ie.any() else 600.0
        fig=plt.figure(figsize=(18,5.2))
        A=geo(fig,1); A.set_global(); A.set_title(f"concept {cc} mean — global")
        A.tricontourf(nlon,nlat,np.clip(val,0,vg),levels=np.linspace(vg*.05,vg,12),cmap=CMAP,extend="max",transform=PC,zorder=2)
        A=geo(fig,2); A.set_extent(ext,crs=PC); A.set_title(f"concept {cc} mean — zoom")
        mp=A.tricontourf(nlon,nlat,np.clip(val,0,vz),levels=np.linspace(vz*.05,vz,12),cmap=CMAP,extend="max",transform=PC,zorder=2)
        if (freq>0.5).any(): A.contour(MLON,MLAT,freq,[0.5],colors="#111",linewidths=1.6,transform=PC,zorder=6)
        A.gridlines(draw_labels=True,lw=.3,color="#bbb"); fig.colorbar(mp,ax=A,shrink=.65,label="mean activation")
        A=geo(fig,3); A.set_extent(ext,crs=PC); A.set_title("mean IVT — zoom")
        ip=A.tricontourf(nlon,nlat,iv,levels=np.linspace(100,vi_,12),cmap="YlGnBu",extend="max",transform=PC,zorder=2)
        if (freq>0.5).any(): A.contour(MLON,MLAT,freq,[0.5],colors="#111",linewidths=1.6,transform=PC,zorder=6)
        A.gridlines(draw_labels=True,lw=.3,color="#bbb"); fig.colorbar(ip,ax=A,shrink=.65,label="IVT kg m$^{-1}$ s$^{-1}$")
        fig.suptitle(f"{r}: AR-specific concept {cc} mean footprint over {n} ARs (black = AR present >50% of ARs)",y=1.03)
        fig.savefig(f"{PLOTS}/footprint_{r}_c{cc}.png",dpi=160,bbox_inches="tight"); plt.close(fig)
        print(r,cc,"saved",flush=True)
    print("FOOTPRINT FIGS DONE")
if __name__=="__main__": main()
