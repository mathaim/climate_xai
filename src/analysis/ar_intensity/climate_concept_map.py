"""Climate-style AR view: IVT magnitude + transport vectors + 250 contour, beside the intensity concept."""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
from src.analysis.ar_intensity.ivt import layer_thickness_pa
import cartopy.crs as ccrs, cartopy.feature as cfeature
PLOTS="results/ar_intensity/plots"; SAE="plain_L8"; G=9.81
CONCEPT={"W_N_America":1592,"W_Europe":2948,"W_S_America":3218,"E_Australia":3720}
conv=lambda x:x-360 if x>180 else x; PC=ccrs.PlateCarree()
CMAP=LinearSegmentedColormap.from_list("moist",["#e8f6f9","#9fdcc4","#4eb3d3","#2b6cb0","#3b3b98","#7d3c98","#c0392b"])
def geo(fig,pos):
    A=fig.add_subplot(1,2,pos,projection=PC); A.add_feature(cfeature.LAND,facecolor="#f2efe9",zorder=0)
    A.add_feature(cfeature.OCEAN,facecolor="#e3edf3",zorder=0); A.coastlines("50m",color="#555",lw=.6,zorder=5); return A
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index(); dp=layer_thickness_pa(levels)
    F,md=load(SAE,"region_magnitude"); reg=md.region.to_numpy(); tix=md.time_index.to_numpy().astype(int)
    m,c,fmin,frng=load_sae(SAE,dev)
    for r in REGIONS:
        cc=CONCEPT[r]; sel=reg==r; ti=int(tix[sel][np.argmax(F[sel][:,cc])]); dt=index_to_datetime(ti)
        era=np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        nlat=era[:,lat_i]; nlon=np.array([conv(x) for x in era[:,lon_i]])
        qu=(era[:,qi]*era[:,ui]*dp[None,:]).sum(1)/G; qv=(era[:,qi]*era[:,vi]*dp[None,:]).sum(1)/G; mag=np.sqrt(qu**2+qv**2)
        a=np.load(act_file(c,dt),mmap_mode="r"); xr=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
        with torch.no_grad(): val=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()[:,cc]
        cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]; xs=[conv(x) for seg in lons for x in seg]
        ext=[min(xs)-15,max(xs)+15,la[0]-10,la[1]+12]
        ie=(nlon>=ext[0])&(nlon<=ext[1])&(nlat>=ext[2])&(nlat<=ext[3]); sub=np.where(ie)[0]; sub=sub[::max(1,len(sub)//350)]
        def box(A):
            for x0,x1 in lons: A.plot([conv(x0),conv(x1),conv(x1),conv(x0),conv(x0)],[la[0],la[0],la[1],la[1],la[0]],c="#c0392b",lw=1.8,transform=PC,zorder=8)
        fig=plt.figure(figsize=(15,6.2))
        A=geo(fig,1); A.set_extent(ext,crs=PC); A.set_title("IVT + moisture transport")
        cf=A.tricontourf(nlon,nlat,mag,levels=[100,250,400,550,700,850,1000,1200],cmap="YlGnBu",extend="max",transform=PC,zorder=2)
        A.tricontour(nlon,nlat,mag,[250],colors="k",linewidths=1.3,transform=PC,zorder=4)
        q=A.quiver(nlon[sub],nlat[sub],qu[sub],qv[sub],transform=PC,scale=2.2e4,width=.0035,color="#222",zorder=6)
        A.quiverkey(q,.88,1.04,500,"500 kg m$^{-1}$s$^{-1}$",labelpos="E",fontproperties={"size":9}); box(A)
        A.gridlines(draw_labels=True,lw=.3); fig.colorbar(cf,ax=A,shrink=.62,label="IVT kg m$^{-1}$ s$^{-1}$")
        A=geo(fig,2); A.set_extent(ext,crs=PC); A.set_title(f"SAE concept {cc} activation")
        vz=float(np.percentile(val[ie&(val>0)],99)) if (ie&(val>0)).any() else 1.0
        mp=A.tricontourf(nlon,nlat,np.clip(val,0,vz),levels=np.linspace(vz*.05,vz,12),cmap=CMAP,extend="max",transform=PC,zorder=2)
        A.tricontour(nlon,nlat,mag,[250],colors="k",linewidths=1.0,linestyles="--",transform=PC,zorder=4); box(A)
        A.gridlines(draw_labels=True,lw=.3); fig.colorbar(mp,ax=A,shrink=.62,label="activation")
        fig.suptitle(f"{r}: AR on {dt:%Y-%m-%d %Hz} — IVT (left) vs intensity concept {cc} (right); dashed = 250 IVT (AR edge)",y=1.02)
        fig.savefig(f"{PLOTS}/climate_{r}_c{cc}.png",dpi=160,bbox_inches="tight"); plt.close(fig)
        print(f"{r}: concept {cc} {dt}",flush=True)
    print("CLIMATE MAPS DONE")
if __name__=="__main__": main()
