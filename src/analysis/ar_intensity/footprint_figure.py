"""Compare AR-concept activation vs ground-truth AR presence: maps + spatial-match correlation."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.ivt_pipeline import NPZ, region_node_setup
import cartopy.crs as ccrs, cartopy.feature as cfeature
OUT="results/ar_intensity/baseline"; PLOTS="results/ar_intensity/plots"; SAE="plain_L8"
conv=lambda x: x-360 if x>180 else x; PC=ccrs.PlateCarree()
CMAP=LinearSegmentedColormap.from_list("moist",["#e8f6f9","#9fdcc4","#4eb3d3","#2b6cb0","#3b3b98","#7d3c98","#c0392b"])
def geo(fig,pos):
    A=fig.add_subplot(1,2,pos,projection=PC); A.add_feature(cfeature.LAND,facecolor="#f0ede6",zorder=0)
    A.add_feature(cfeature.OCEAN,facecolor="#dce7ef",zorder=0); A.coastlines("110m",color="#555",lw=.6,zorder=4); return A
def main():
    z=np.load(f"{OUT}/mean_footprint_{SAE}.npz"); d=np.load(NPZ); setup=region_node_setup(); grid_lon=np.arange(0,360,0.25)
    nlat=z["nlat"]; nlon=np.array([conv(x) for x in z["nlon"]]); scores=[]
    for r in REGIONS:
        cc=int(z[f"{r}_concept"][0]); val=z[f"{r}_act"]; freq0=z[f"{r}_freq"]; n=int(z[f"{r}_n"][0])
        s=setup[r]; an=val[s["nodes"]]; arn=freq0[s["li"],s["ji"]]
        rm=float(np.corrcoef(an,arn)[0,1]) if an.std()>0 and arn.std()>0 else float("nan")
        scores.append((r,cc,rm))
        cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]; xs=[conv(x) for seg in lons for x in seg]
        ext=[min(xs)-4,max(xs)+4,la[0]-3,la[1]+3]
        mlat=d[f"{r}__lat"]; mlon=np.array([conv(x) for x in np.concatenate([grid_lon[(grid_lon>=x)&(grid_lon<=y)] for x,y in lons])])
        o=np.argsort(mlon); mlon=mlon[o]; freq=freq0[:,o]; MLON,MLAT=np.meshgrid(mlon,mlat)
        ie=(nlon>=ext[0])&(nlon<=ext[1])&(nlat>=ext[2])&(nlat<=ext[3])
        vz=float(np.percentile(val[ie&(val>0)],99)) if (ie&(val>0)).any() else 1.0
        fig=plt.figure(figsize=(13,5))
        A=geo(fig,1); A.set_extent(ext,crs=PC); A.set_title(f"AR-concept {cc} mean activation")
        mp=A.tricontourf(nlon,nlat,np.clip(val,0,vz),levels=np.linspace(vz*.05,vz,12),cmap=CMAP,extend="max",transform=PC,zorder=2)
        if (freq>0.5).any(): A.contour(MLON,MLAT,freq,[0.5],colors="#111",linewidths=1.4,transform=PC,zorder=6)
        A.gridlines(draw_labels=True,lw=.3,color="#bbb"); fig.colorbar(mp,ax=A,shrink=.6,label="mean activation")
        A=geo(fig,2); A.set_extent(ext,crs=PC); A.set_title("ground truth: AR occurrence frequency")
        cf=A.contourf(MLON,MLAT,freq,levels=np.linspace(0,max(freq.max(),.01),11),cmap="YlGnBu",transform=PC,zorder=2)
        A.gridlines(draw_labels=True,lw=.3,color="#bbb"); fig.colorbar(cf,ax=A,shrink=.6,label="fraction of ARs")
        fig.suptitle(f"{r}: concept {cc} vs AR presence — spatial match r={rm:.2f} (n={n} ARs)",y=1.02)
        fig.savefig(f"{PLOTS}/footprint_{r}_c{cc}.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print(f"{'region':13} {'concept':>7} {'AR-match r':>10}")
    for r,cc,rm in scores: print(f"{r:13} {cc:7d} {rm:10.2f}")
    print("FOOTPRINT FIGS DONE")
if __name__=="__main__": main()
