"""Per-season concept-vs-region-IVT correlation, every year 1979-2017 (existing data, no streaming)."""
import os, numpy as np, pandas as pd, torch
from datetime import datetime, timedelta
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, region_node_setup, node_ivt, ERA5_DIR
SAE="plain_L8"; OUT="results/ar_intensity/baseline"
CONCEPT={"W_N_America":1592,"W_Europe":2948,"W_S_America":3218,"E_Australia":3720}
NH={"W_N_America","W_Europe"}
def window(r,Y):  # NH winter DJF (Dec Y - Feb Y+1); SH winter JJA (Jun-Aug Y)
    return (datetime(Y,12,1),datetime(Y+1,3,1)) if r in NH else (datetime(Y,6,1),datetime(Y,9,1))
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index(); setup=region_node_setup()
    m,c,fmin,frng=load_sae(SAE,dev); rows=[]
    for r in REGIONS:
        cc=CONCEPT[r]; nodes=setup[r]["nodes"]; yrs=range(1979,2017) if r in NH else range(1979,2018)
        for Y in yrs:
            s,e=window(r,Y); A=[]; IV=[]; dt=s
            while dt<e:
                try:
                    a=np.load(act_file(c,dt),mmap_mode="r"); xr=np.ascontiguousarray(a[nodes]).astype(np.float32).reshape(len(nodes),-1)
                    era=np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
                    with torch.no_grad(): av=float(encode(m,c["arch"],torch.from_numpy(xr).to(dev)).cpu().numpy()[:,cc].sum())
                    A.append(av); IV.append(float(node_ivt(era[nodes],qi,ui,vi,levels).max()))
                except Exception: pass
                dt=dt+timedelta(hours=6)
            if len(A)>20 and np.std(A)>0 and np.std(IV)>0:
                rows.append(dict(region=r,year=Y,r=float(np.corrcoef(A,IV)[0,1]),n=len(A)))
        print(r,"done",len([x for x in rows if x['region']==r]),"seasons",flush=True)
    df=pd.DataFrame(rows); df.to_csv(f"{OUT}/timeseries_yearly_r.csv",index=False)
    print(df.groupby("region").r.agg(["count","mean","std","min","max"]).round(3))
    print("SAVED timeseries_yearly_r.csv")
if __name__=="__main__": main()
