"""Concept activation vs region max-IVT as time series over a fixed wet-season window."""
import numpy as np, torch
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime, AR_START
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, region_node_setup, node_ivt, ERA5_DIR
PLOTS="results/ar_intensity/plots"; SAE="plain_L8"
CONCEPT={"W_N_America":1592,"W_Europe":2948,"W_S_America":3218,"E_Australia":3720}
WIN={"W_N_America":(datetime(2015,12,1),datetime(2016,2,28)),
     "W_Europe":(datetime(2015,12,1),datetime(2016,2,28)),
     "W_S_America":(datetime(2015,6,1),datetime(2015,8,31)),
     "E_Australia":(datetime(2015,6,1),datetime(2015,8,31))}
def dti(dt): return int(round((dt-AR_START).total_seconds()/3600/6))+1
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index(); setup=region_node_setup()
    m,c,fmin,frng=load_sae(SAE,dev)
    fig,axes=plt.subplots(len(REGIONS),1,figsize=(13,3.2*len(REGIONS))); scores=[]
    for ax,r in zip(np.atleast_1d(axes),REGIONS):
        cc=CONCEPT[r]; nodes=setup[r]["nodes"]; lo=dti(WIN[r][0]); hi=dti(WIN[r][1])
        T=[]; A=[]; IV=[]
        for ti in range(lo,hi+1):
            dt=index_to_datetime(ti)
            try:
                a=np.load(act_file(c,dt),mmap_mode="r"); xr=np.ascontiguousarray(a[nodes]).astype(np.float32).reshape(len(nodes),-1)
                era=np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
            except Exception: continue
            with torch.no_grad(): av=float(encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()[:,cc].sum())
            T.append(dt); A.append(av); IV.append(float(node_ivt(era[nodes],qi,ui,vi,levels).max()))
        A=np.array(A); IV=np.array(IV); rr=float(np.corrcoef(A,IV)[0,1]) if len(A)>2 else float("nan"); scores.append((r,cc,rr))
        ax.plot(T,A,color="#c0392b",lw=1.4,label=f"concept {cc}")
        ax2=ax.twinx(); ax2.plot(T,IV,color="#185FA5",lw=1.4,alpha=.75,label="region max IVT")
        ax.set_ylabel("concept activation",color="#c0392b"); ax2.set_ylabel("max IVT (kg/m/s)",color="#185FA5")
        ax.set_title(f"{r}: concept {cc} vs region max-IVT — r={rr:.2f}  ({WIN[r][0]:%Y-%m}..{WIN[r][1]:%Y-%m})")
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/timeseries_concept_ivt.png",dpi=150,bbox_inches="tight")
    print(f"\n{'region':13} {'concept':>7} {'time r':>7}")
    for r,cc,rr in scores: print(f"{r:13} {cc:7d} {rr:7.2f}")
    print("TIMESERIES DONE")
if __name__=="__main__": main()
