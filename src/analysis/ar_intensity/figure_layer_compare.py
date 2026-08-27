"""Concept 340 firing locations, single SH strip; title above, colorbar below. Western South America
zoom is an inset over the empty South Atlantic with leader lines auto-linking the facing edge of its
source box. Left edge cropped to drop the empty Pacific. Count-weighted auto-zoom.
Env: QLO/QHI, MLAT/MLON, NORTH, LATMIN/LATMAX/LONMIN/LONMAX, SG; IX0/IX1/IY0/IY1 (inset host, deg)."""
import numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
def envf(k,d):
    v=os.environ.get(k,""); return float(v) if v!="" else d
D8=np.load("/scratch/euh7ys/climate_xai/concept_ivt/characterize_wsa_8k.npz")
nlat=D8["nlat"].astype(float); nlon=((D8["nlon"].astype(float)+180)%360)-180; PC=ccrs.PlateCarree()
def cnt(cc): return D8[f"cnt_{cc}"]
c=cnt(340); f=c>0; la=nlat[f]; lo=nlon[f]; wt=c[f].astype(float)
def wpct(v,w,q):
    o=np.argsort(v); vs=v[o]; ws=w[o]; cw=np.cumsum(ws)/ws.sum(); return float(np.interp(q/100.0,cw,vs))
QLO=envf("QLO",1.0); QHI=envf("QHI",99.0); MLAT=envf("MLAT",4.0); MLON=envf("MLON",6.0); NORTH=envf("NORTH",8.0)
latmin=max(-89,envf("LATMIN",wpct(la,wt,QLO)-MLAT)); latmax=min(89,envf("LATMAX",wpct(la,wt,QHI)+MLAT+NORTH))
lonmin=max(-180,envf("LONMIN",-100.0)); lonmax=min(180,envf("LONMAX",wpct(lo,wt,QHI)+MLON))
GEXT=[lonmin,lonmax,latmin,latmax]; A=(GEXT[1]-GEXT[0])/(GEXT[3]-GEXT[2])
print("global extent:",[round(float(x),1) for x in GEXT],"aspect",round(A,2))
EXT=[-83,-58,-57,-27]; inreg=(nlon>=EXT[0])&(nlon<=EXT[1])&(nlat>=EXT[2])&(nlat<=EXT[3])
COL=["#d62728","#ff7f0e","#2ca02c"]; LAB=["3481","3948","3675"]; OUT340="#2c7fb8"; SG=envf("SG",20.0)
figW=16.0; mapw=0.98; map_w_in=mapw*figW; map_h_in=map_w_in/A; title_in=0.6; cbar_in=0.8
figH=map_h_in+title_in+cbar_in; fig=plt.figure(figsize=(figW,figH)); print("figsize",round(figW,2),round(figH,2))
gx=(1-mapw)/2; gw=mapw; gy=cbar_in/figH; gh=map_h_in/figH
axG=fig.add_axes([gx,gy,gw,gh],projection=PC); axG.set_extent(GEXT,crs=PC)
axG.add_feature(cfeature.LAND,facecolor="0.9",zorder=0); axG.coastlines(lw=0.5,color="0.45",zorder=4)
vmaxG=float(np.quantile(c[f],0.98))
scG=axG.scatter(nlon[f],nlat[f],c=c[f],s=SG,marker="s",cmap="YlOrRd",norm=Normalize(0,vmaxG),edgecolor="none",transform=PC,zorder=2)
axG.add_patch(Rectangle((EXT[0],EXT[2]),EXT[1]-EXT[0],EXT[3]-EXT[2],fill=False,edgecolor="k",lw=1.4,ls="--",transform=PC,zorder=6))
fig.text(0.5, gy+gh+0.45*(title_in/figH), "Firing Locations of Concept 340 & Children", ha="center", va="center", fontsize=22, fontweight="bold")
def to_fig(lon,lat): return (gx+gw*(lon-GEXT[0])/(GEXT[1]-GEXT[0]), gy+gh*(lat-GEXT[2])/(GEXT[3]-GEXT[2]))
cbw=0.34; cax=fig.add_axes([0.5-cbw/2, 0.50*gy, cbw, 0.17*gy])
cb=fig.colorbar(scG,cax=cax,orientation="horizontal"); cb.set_label("active timesteps",fontsize=16,fontweight="bold"); cb.ax.tick_params(labelsize=13)
ix0g=envf("IX0",-52.0); ix1g=envf("IX1",12.0); iy0g=envf("IY0",-62.0); iy1g=envf("IY1",-18.0)
hx0,hy0=to_fig(ix0g,iy0g); hx1,hy1=to_fig(ix1g,iy1g); hw=hx1-hx0; hh=hy1-hy0
wsa_asp=(EXT[1]-EXT[0])/(EXT[3]-EXT[2]); ih=hh*0.96; iw=ih*wsa_asp*figH/figW
if iw>hw: iw=hw*0.96; ih=iw/(wsa_asp*figH/figW)
ix=hx0+(hw-iw)/2; iy=hy0+(hh-ih)/2
axC=fig.add_axes([ix,iy,iw,ih],projection=PC); axC.set_extent(EXT,crs=PC)
axC.add_feature(cfeature.LAND,facecolor="0.97",zorder=0); axC.coastlines("50m",lw=0.9,zorder=4)
for s in axC.spines.values(): s.set_edgecolor("k"); s.set_linewidth(1.2)
axC.scatter(nlon[inreg],nlat[inreg],s=22,marker="s",facecolor="white",edgecolor="0.85",lw=0.3,transform=PC,zorder=1)
c340=cnt(340); f340=(c340>0)&inreg
axC.scatter(nlon[f340],nlat[f340],s=26,marker="s",facecolor="none",edgecolor=OUT340,lw=1.1,transform=PC,zorder=2,label="parent 340")
for k,col,l in zip([3481,3948,3675],COL,LAB):
    cn=cnt(k); fr=(cn>0)&inreg
    axC.scatter(nlon[fr],nlat[fr],s=22,marker="s",c=col,alpha=0.9,edgecolor="none",transform=PC,zorder=3,label=l)
axC.add_patch(Rectangle((-77,-50),15,20,fill=False,edgecolor="k",lw=1.2,ls="--",transform=PC,zorder=6))
axC.legend(fontsize=10,loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=2,frameon=True,facecolor="white",edgecolor="0.7",framealpha=0.92,columnspacing=1.1,handletextpad=0.4)
s_cx=0.5*(to_fig(EXT[0],EXT[2])[0]+to_fig(EXT[1],EXT[2])[0]); i_cx=ix+iw/2
xe,iedge=(EXT[1],ix) if i_cx>=s_cx else (EXT[0],ix+iw)
for sa,ib in [(to_fig(xe,EXT[2]),(iedge,iy)),(to_fig(xe,EXT[3]),(iedge,iy+ih))]:
    fig.add_artist(Line2D([sa[0],ib[0]],[sa[1],ib[1]],color="0.45",lw=0.8,zorder=8))
fig.savefig("/scratch/euh7ys/climate_xai/plots/layer_compare.png",dpi=170,bbox_inches="tight"); print("saved")
