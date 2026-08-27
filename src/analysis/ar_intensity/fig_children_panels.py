"""Three global children of concept 99 with the parent's presence mask (pale blue) as backdrop.
Shared active-timesteps scale + one colorbar (PERPANEL=1 for per-panel). Manual 2:1 axes placement
so the maps fill their boxes with no letterbox whitespace. Row title on top; per-panel titles centered."""
import numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cartopy.crs as ccrs, cartopy.feature as cfeature
D=np.load("/scratch/euh7ys/climate_xai/concept_ivt/footprint_99_8k.npz")
OUT="/scratch/euh7ys/climate_xai/plots"; PC=ccrs.PlateCarree()
conv=lambda x: ((np.asarray(x,float)+180)%360)-180
nlon=conv(D["nlon"]); nlat=D["nlat"].astype(float)
CH=[3392,1454,2722]
PKEY=next((k for k in ["cnt_99","cnt_0099","cnt_parent","cnt_99_0","p99"] if k in D.files), None)
P99="#c6dbef"
CMAP=LinearSegmentedColormap.from_list("c",["#f7f4ef","#fed976","#fd8d3c","#e31a1c","#800026"])
PERPANEL=os.environ.get("PERPANEL","")=="1"; SHARED=not PERPANEL
if SHARED:
    allc=np.concatenate([D[f"cnt_{cc}"].astype(float)[D[f"cnt_{cc}"]>0] for cc in CH]); VMAX=float(np.percentile(allc,99.5))
figW=18.0; figH=4.9; fig=plt.figure(figsize=(figW,figH))
LEFT=0.01; G=0.012; MW=(0.98-2*G)/3; MH=(MW*figW/2.0)/figH; Y0=0.15
axes=[]
for i,cc in enumerate(CH):
    ax=fig.add_axes([LEFT+i*(MW+G),Y0,MW,MH],projection=PC); axes.append(ax)
    ax.add_feature(cfeature.LAND,facecolor="#eeeae0",zorder=0)
    ax.add_feature(cfeature.COASTLINE,lw=0.4,edgecolor="0.45",zorder=4); ax.set_global()
    if PKEY is not None:
        p=D[PKEY].astype(float); mp=p>0
        ax.scatter(nlon[mp],nlat[mp],s=6,c=P99,edgecolor="none",rasterized=True,transform=PC,zorder=1)
    c=D[f"cnt_{cc}"].astype(float); m=c>0; o=np.argsort(c[m])
    vmax=VMAX if SHARED else float(np.percentile(c[m],99.5)); nl=Normalize(0,vmax)
    sc=ax.scatter(nlon[m][o],nlat[m][o],s=6,c=c[m][o],cmap=CMAP,norm=nl,edgecolor="none",rasterized=True,transform=PC,zorder=2)
    ax.set_title(f"Concept {cc}",fontsize=16,loc="center",fontweight="bold")
    if not SHARED:
        cax=fig.add_axes([LEFT+i*(MW+G)+MW*0.15,Y0-0.11,MW*0.7,0.028])
        cb=fig.colorbar(sc,cax=cax,orientation="horizontal"); cb.set_label("active timesteps",fontsize=12,fontweight="bold"); cb.ax.tick_params(labelsize=10)
if SHARED:
    cax=fig.add_axes([0.5-0.16,0.07,0.32,0.03])
    cb=fig.colorbar(sc,cax=cax,orientation="horizontal"); cb.set_label("active timesteps",fontsize=14,fontweight="bold"); cb.ax.tick_params(labelsize=12)
fig.suptitle("Firing Locations of Concept 99 & Children",fontsize=23,fontweight="bold",y=0.92)
fig.savefig(f"{OUT}/ar_children_panels.png",dpi=170,bbox_inches="tight")
print("saved ar_children_panels.png")
