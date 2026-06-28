"""Composite concept-activation maps for low- vs high-intensity AR events, per region.
Shows whether the concept's spatial footprint strengthens/broadens as IVT rises."""
import glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
from src.analysis.ar_intensity import concept_ivt_core as C
import cartopy.crs as ccrs, cartopy.feature as cfeature
PLOTS="/scratch/euh7ys/climate_xai/plots"; TRACK="/scratch/euh7ys/climate_xai/concept_ivt"
SAE="plain_L8"; NEV=50
CONCEPT={"W_N_America":1592,"W_Europe":2948,"W_S_America":3218,"E_Australia":816}
conv=lambda x:x-360 if x>180 else x; PC=ccrs.PlateCarree()
CMAP=LinearSegmentedColormap.from_list("moist",["#e8f6f9","#9fdcc4","#4eb3d3","#2b6cb0","#3b3b98","#7d3c98","#c0392b"])
def geo(fig,pos,n):
    A=fig.add_subplot(1,n,pos,projection=PC); A.add_feature(cfeature.LAND,facecolor="#f2efe9",zorder=0)
    A.add_feature(cfeature.OCEAN,facecolor="#e3edf3",zorder=0); A.coastlines("50m",color="#555",lw=.6,zorder=5); return A
def composite(tis,cc,ie_idx,m,c,dev):
    acc=None; n=0
    for ti in tis:
        dt=index_to_datetime(int(ti))
        try: a=np.load(act_file(c,dt),mmap_mode="r")
        except Exception: continue
        xr=np.ascontiguousarray(a[ie_idx]).astype(np.float32).reshape(len(ie_idx),-1)
        with torch.no_grad(): val=encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()[:,cc]
        acc=val if acc is None else acc+val; n+=1
    return acc/max(n,1), n
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
    m,c,fmin,frng=load_sae(SAE,dev)
    era0=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat=era0[:,lat_i]; nlon=np.array([conv(x) for x in era0[:,lon_i]]); rng=np.random.default_rng(0)
    for r in REGIONS:
        cc=CONCEPT[r]; cfg=REGIONS[r]; la=cfg["lat"]; lons=cfg["lon"]
        xs=[conv(x) for seg in lons for x in seg]; ext=[min(xs)-15,max(xs)+15,la[0]-10,la[1]+12]
        ie_idx=np.where((nlon>=ext[0])&(nlon<=ext[1])&(nlat>=ext[2])&(nlat<=ext[3]))[0]
        d=np.load(f"{TRACK}/track_pool_{r}.npz"); ivt=d["ivt"].astype(float); tindex=d["tindex"]; del d
        ok=np.isfinite(ivt); ivt=ivt[ok]; ti=tindex[ok]; reg=C.ivt_regime(ivt)
        hi_iv=ivt[reg=="intense"]; hi_all=ti[reg=="intense"]; hi_ti=hi_all[np.argsort(hi_iv)[::-1][:NEV]]
        wk_pool=ti[reg=="weak"]; wk_ti=rng.choice(wk_pool,min(NEV,len(wk_pool)),replace=False)
        lo,nl=composite(wk_ti,cc,ie_idx,m,c,dev); hi,nh=composite(hi_ti,cc,ie_idx,m,c,dev)
        nlo=nlon[ie_idx]; nla=nlat[ie_idx]; diff=hi-lo
        vmax=float(np.percentile(np.concatenate([lo,hi]),99)); dmax=float(np.percentile(np.abs(diff),99)) or 1e-3
        def box(A):
            for x0,x1 in lons: A.plot([conv(x0),conv(x1),conv(x1),conv(x0),conv(x0)],[la[0],la[0],la[1],la[1],la[0]],c="#111",lw=1.6,transform=PC,zorder=8)
        fig=plt.figure(figsize=(18,5.5))
        panels=[("Low intensity (weak AR)",lo,CMAP,0.0,vmax),("High intensity (top 10%)",hi,CMAP,0.0,vmax),("High - Low",diff,"RdBu_r",-dmax,dmax)]
        for pos,(lab,field,cm,vmn,vmx) in enumerate(panels,1):
            A=geo(fig,pos,3); A.set_extent(ext,crs=PC); A.set_title(lab)
            tcf=A.tricontourf(nlo,nla,field,levels=np.linspace(vmn,vmx,13),cmap=cm,extend="both",transform=PC,zorder=2)
            box(A); A.gridlines(draw_labels=True,lw=.3); fig.colorbar(tcf,ax=A,shrink=.6)
        fig.suptitle(f"{r}: concept {cc} composite activation by AR intensity (n_low={nl}, n_high={nh})",y=1.02)
        fig.savefig(f"{PLOTS}/composite_{r}_c{cc}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        print(f"{r}: c{cc} low_n={nl} high_n={nh} vmax={vmax:.3f}",flush=True)
    print("COMPOSITES DONE")
if __name__=="__main__": main()
