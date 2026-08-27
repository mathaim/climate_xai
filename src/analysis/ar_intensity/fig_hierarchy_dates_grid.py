"""4x3: row0 = IVT ground truth, rows1-3 = layers (AR core 139/99/111); cols = 3 strongest global-AR dates.
Core concept only. Dates auto-picked by L8 core-99 total firing over 24 evenly-spaced candidates (distinct years)."""
import os, glob, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR, node_ivt
DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT="/scratch/euh7ys/climate_xai/plots"; conv=lambda x: x-360 if x>180 else x; THRESH=0.1; PC=ccrs.PlateCarree()
CORE={"matry_L0":139,"matry_L8":99,"matry_L15":111}
LAYERS=[("matry_L0","layer 0"),("matry_L8","layer 8"),("matry_L15","layer 15")]
CMAP=LinearSegmentedColormap.from_list("m",["#bfe3ee","#4eb3d3","#2b6cb0","#3b3b98","#7d3c98","#c0392b"])
NL=Normalize(0.0,0.5)  # shared core-activation scale for all three layer rows
idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
era0=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
NLAT=era0[:,lat_i].astype(float); NLON=np.array([conv(x) for x in era0[:,lon_i].astype(float)])
def dstr(fn): return fn.split("_t")[-1].replace(".npy","")
def enc(m,c,fmin,frng,f,cc):
    a=np.load(f,mmap_mode="r"); x=np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    if fmin is not None: x=(2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m,c["arch"],torch.from_numpy(x).to(DEV)).cpu().numpy()[:,cc]
def rowlab(ax,t): ax.text(-0.06,0.5,t,transform=ax.transAxes,rotation=90,va="center",ha="center",fontsize=12,fontweight="bold")
print("device",DEV,flush=True)
m8,c8,fm8,fr8=load_sae("matry_L8",DEV); f8=sorted(glob.glob(f"{c8['act']}/layer*_*.npy"))
DATES=os.environ.get("DATES","").strip()
if DATES:
    picked=sorted(DATES.split(","))
else:
    cand=[f8[i] for i in np.linspace(0,len(f8)-1,48).astype(int)]; sc=[]
    for f in cand:
        n=float((enc(m8,c8,fm8,fr8,f,99)>THRESH).sum()); sc.append((n,dstr(f))); print("cand",dstr(f),int(n),flush=True)
    sc.sort(reverse=True)
    def third(ds):
        y=int(ds[:4]); return 0 if y<1992 else (1 if y<2005 else 2)
    pbt={}
    for n,ds in sc:
        t=third(ds)
        if t not in pbt: pbt[t]=ds
        if len(pbt)==3: break
    picked=sorted(pbt.values())
print("PICKED",picked,flush=True)
fig=plt.figure(figsize=(16,9.2)); gs=fig.add_gridspec(4,3,wspace=0.02,hspace=0.01)
# row 0: IVT ground truth
ivts=[node_ivt(np.load(f"{ERA5_DIR}/era5_inputs_{ds}.npy"),qi,ui,vi,levels) for ds in picked]
IVTD=dict(zip(picked,ivts)); STAT=[]
nrm=Normalize(100,float(np.quantile(np.concatenate(ivts),0.98))); axs=[]
for ci,(ds,iv) in enumerate(zip(picked,ivts)):
    ax=fig.add_subplot(gs[0,ci],projection=PC); axs.append(ax)
    ax.add_feature(cfeature.LAND,facecolor="#f4f1ea",zorder=0); ax.add_feature(cfeature.COASTLINE,lw=0.4,edgecolor="0.45"); ax.set_global()
    ax.scatter(NLON,NLAT,s=2.5,c=iv,cmap="YlGnBu",norm=nrm,edgecolor="none",transform=PC,rasterized=True)
    ax.set_title(ds,fontsize=13,fontweight="bold")
    if ci==0: rowlab(ax,"IVT ground truth")
fig.colorbar(ScalarMappable(norm=nrm,cmap="YlGnBu"),ax=axs,location="right",shrink=0.85,pad=0.008,label="IVT kg m$^{-1}$ s$^{-1}$")
# rows 1-3: layers (shared 0-0.5 scale)
FIRE={}; layer_axs=[]
for ri0,(sae,lab) in enumerate(LAYERS):
    ri=ri0+1; m,c,fmin,frng=load_sae(sae,DEV); cc=CORE[sae]; V={}
    for ds in picked:
        g=glob.glob(f"{c['act']}/*_t{ds}.npy"); V[ds]=enc(m,c,fmin,frng,g[0],cc) if g else np.zeros(len(NLAT),np.float32)
    FIRE[sae]={ds:(V[ds]>THRESH) for ds in picked}
    for ci,ds in enumerate(picked):
        v=V[ds]; ax=fig.add_subplot(gs[ri,ci],projection=PC); layer_axs.append(ax)
        ax.add_feature(cfeature.LAND,facecolor="#f4f1ea",zorder=0); ax.add_feature(cfeature.COASTLINE,lw=0.4,edgecolor="0.45"); ax.set_global()
        fire=v>THRESH
        if fire.any(): ax.scatter(NLON[fire],NLAT[fire],s=2.5,c=v[fire],cmap=CMAP,norm=NL,edgecolor="none",transform=PC,rasterized=True)
        if ci==0: rowlab(ax,f"{lab}\nconcept {cc}")
        # stats for this (layer, date)
        iv=IVTD[ds]; r=float(np.corrcoef(v,iv)[0,1]); nfire=int(fire.sum())
        STAT.append((lab,cc,ds,nfire,float(v[fire].mean()) if fire.any() else 0.0,
                     float(v.max()),float(iv[fire].mean()) if fire.any() else 0.0,r))
fig.colorbar(ScalarMappable(norm=NL,cmap=CMAP),ax=layer_axs,location="right",shrink=0.6,pad=0.008,label="concept activation")
# cross-layer Jaccard of active sets (averaged over dates)
def jac(a,b): u=(a|b).sum(); return float((a&b).sum()/u) if u else 0.0
print("\n=== per (layer,date) stats: nfire, mean act, peak act, mean IVT at fire, corr(act,IVT) ===",flush=True)
for lab,cc,ds,nf,ma,pk,mi,r in STAT:
    print(f"  {lab:8} c{cc} {ds}: nfire={nf:5d} mean={ma:.3f} peak={pk:.3f} IVT@fire={mi:6.1f} corr(IVT)={r:+.3f}",flush=True)
print("\n=== cross-layer spatial overlap (Jaccard of active nodes, mean over dates) ===",flush=True)
saes=[s for s,_ in LAYERS]
for i in range(len(saes)):
    for j in range(i+1,len(saes)):
        js=np.mean([jac(FIRE[saes[i]][ds],FIRE[saes[j]][ds]) for ds in picked])
        print(f"  {saes[i]} vs {saes[j]}: J={js:.3f}",flush=True)
fig.savefig(f"{OUT}/hierarchy_dates_grid.png",dpi=160,bbox_inches="tight"); print("saved hierarchy_dates_grid.png",flush=True)
