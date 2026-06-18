"""AR presence on concept-active timesteps (no re-encode): mask averaged, weighted by concept activation."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity._load import load
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.ivt_pipeline import NPZ
import cartopy.crs as ccrs, cartopy.feature as cfeature
OUT="results/ar_intensity/baseline"; PLOTS="results/ar_intensity/plots"; SAE="plain_L8"
CONCEPT={"W_N_America":3069,"W_Europe":1008,"W_S_America":1115,"E_Australia":975}
conv=lambda x:x-360 if x>180 else x; PC=ccrs.PlateCarree()
def geo(fig,pos):
    A=fig.add_subplot(1,2,pos,projection=PC); A.add_feature(cfeature.LAND,facecolor="#f0ede6",zorder=0)
    A.add_feature(cfeature.OCEAN,facecolor="#dce7ef",zorder=0); A.coastlines("110m",color="#555",lw=.6,zorder=4); return A
def main():
    d=np.load(NPZ); grid_lon=np.arange(0,360,0.25)
    F,md=load(SAE,"region_magnitude"); reg=md.region.to_numpy(); tix=md.time_index.to_numpy().astype(int)
    for r in REGIONS:
        sel=reg==r; tis=tix[sel]; w=F[sel][:,CONCEPT[r]]
        M=d[f"{r}__mask"][tis-1].astype(float)
        base=M.mean(0); wt=(M*w[:,None,None]).sum(0)/max(w.sum(),1e-9)
        cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]; xs=[conv(x) for seg in lons for x in seg]
        ext=[min(xs)-4,max(xs)+4,la[0]-3,la[1]+3]
        mlat=d[f"{r}__lat"]; mlon=np.array([conv(x) for x in np.concatenate([grid_lon[(grid_lon>=x)&(grid_lon<=y)] for x,y in lons])])
        o=np.argsort(mlon); mlon=mlon[o]; base=base[:,o]; wt=wt[:,o]; MLON,MLAT=np.meshgrid(mlon,mlat)
        vmax=max(base.max(),wt.max(),.01)
        fig=plt.figure(figsize=(13,5))
        for pos,(fld,t) in enumerate([(base,"AR freq — all region ARs"),(wt,f"AR freq weighted by concept {CONCEPT[r]}")],1):
            A=geo(fig,pos); A.set_extent(ext,crs=PC)
            cf=A.contourf(MLON,MLAT,fld,levels=np.linspace(0,vmax,11),cmap="YlGnBu",transform=PC,zorder=2)
            A.gridlines(draw_labels=True,lw=.3,color="#bbb"); A.set_title(t); fig.colorbar(cf,ax=A,shrink=.6,label="fraction")
        fig.suptitle(f"{r}: AR presence on concept-{CONCEPT[r]}-active timesteps (n={int(sel.sum())} ARs)",y=1.02)
        fig.savefig(f"{PLOTS}/arpresence_{r}_c{CONCEPT[r]}.png",dpi=160,bbox_inches="tight"); plt.close(fig)
        print(r,"saved",flush=True)
    print("AR PRESENCE DONE")
if __name__=="__main__": main()
