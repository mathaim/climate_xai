"""
Per-timestep field extractor for the circular-shift null. For a contiguous multi-year window,
store per region: A (T, n_nodes, 512) packed active-bits of the 4096 latents, M (T, n_nodes) AR mask,
node list, and the time index. Ordered in time so M can be circularly shifted. Self-checks IVT@AR.
Env: SAE, YEARS (e.g. 2010-2017). SLURM GPU.
"""
import os, glob, datetime as DT
import numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.regions import REGIONS, AR_START
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup, load_channel_index, node_ivt, NPZ, ERA5_DIR

SAE=os.environ.get("SAE","plain_L8"); Y0,Y1=[int(x) for x in os.environ.get("YEARS","2010-2017").split("-")]
OUTDIR=f"/scratch/euh7ys/climate_xai/concept_ivt/pertimestep_{SAE}"
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("SAE",SAE,"dev",dev,"years",Y0,Y1,flush=True)
    setup=region_node_setup(); d=np.load(NPZ); masks={r:d[f"{r}__mask"] for r in REGIONS}; Tm={r:masks[r].shape[0] for r in REGIONS}
    unodes=np.unique(np.concatenate([setup[r]["nodes"] for r in REGIONS])); posmap={int(x):i for i,x in enumerate(unodes)}
    ridx={r:np.array([posmap[int(x)] for x in setup[r]["nodes"]]) for r in REGIONS}
    idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index(); m,c,fmin,frng=load_sae(SAE,dev)
    sel=[]
    for f in sorted(glob.glob(f"{c['act']}/layer*_*.npy")):
        dt=pdt(os.path.basename(f))
        if Y0<=dt.year<=Y1:
            ti=int(round((dt-AR_START).total_seconds()/21600))+1
            if all(0<=ti-1<Tm[r] for r in REGIONS): sel.append((ti,dt,f))
    sel.sort(); T=len(sel); print(f"{T} timesteps; region nodes:",{r:len(setup[r]['nodes']) for r in REGIONS},flush=True)
    A={r:np.zeros((T,len(setup[r]['nodes']),512),np.uint8) for r in REGIONS}
    M={r:np.zeros((T,len(setup[r]['nodes'])),np.uint8) for r in REGIONS}
    tindex=np.array([s[0] for s in sel],np.int32); checked=False
    for k,(ti,dt,f) in enumerate(sel):
        a=np.load(f,mmap_mode="r"); xr=np.ascontiguousarray(a[unodes]).astype(np.float32).reshape(len(unodes),-1)
        x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
        with torch.no_grad(): acts=(encode(m,c["arch"],torch.from_numpy(x).to(dev))>0).cpu().numpy()
        if not checked:
            era=np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy",mmap_mode="r")
            iv=np.asarray(node_ivt(np.ascontiguousarray(era[unodes]),qi,ui,vi,levels))
        for r in REGIONS:
            lab=(masks[r][ti-1][setup[r]["li"],setup[r]["ji"]]>0)
            M[r][k]=lab.astype(np.uint8); A[r][k]=np.packbits(acts[ridx[r]],axis=-1)
            if not checked and lab.any() and (~lab).any():
                ivr=iv[ridx[r]]; print(f"  [CHECK {r}] IVT@AR={ivr[lab].mean():.0f} IVT@noAR={ivr[~lab].mean():.0f} ARfrac={lab.mean():.2f}",flush=True)
        checked=True
        if (k+1)%1000==0: print(f"  {k+1}/{T}",flush=True)
    os.makedirs(OUTDIR,exist_ok=True)
    for r in REGIONS:
        np.savez(f"{OUTDIR}/{r}.npz",A=A[r],M=M[r],nodes=setup[r]['nodes'],tindex=tindex)
        print(f"saved {r}: A{A[r].shape} ARfrac={M[r].mean():.3f}",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()
