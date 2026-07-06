#!/usr/bin/env python3
"""Per-group concept stats for a per-sample Matryoshka SAE.
Columns: concept range | mean nodes/timestep (group-avg active-node count per
latent) | dead concepts. Processes full timesteps (one file = one timestep)."""
import argparse, glob, os
import numpy as np, torch
from src.models.matryoshka_sae import MatryoshkaSAE
GROUPS=[(0,256,"General"),(256,512,"Level 2"),(512,1024,"Level 3"),
        (1024,2048,"Level 4"),(2048,4096,"Specific")]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layer",type=int,required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--data_dir",required=True)
    ap.add_argument("--n_files",type=int,default=100)
    ap.add_argument("--batch_size",type=int,default=8192)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(a.checkpoint,map_location=dev)
    st=ck["model_state_dict"] if isinstance(ck,dict) and "model_state_dict" in ck else ck
    m=MatryoshkaSAE(512,4096,[256,512,1024,2048,4096],32,300000,topk_mode="per_sample")
    m.load_state_dict(st,strict=False); m.to(dev).eval()
    fmin=np.load(os.path.join(a.data_dir,"feature_min.npy")).astype(np.float32)
    fmax=np.load(os.path.join(a.data_dir,"feature_max.npy")).astype(np.float32)
    frng=fmax-fmin; frng[frng<1e-8]=1.0
    pre=f"layer{a.layer:04d}_mesh_gnn_post_res"
    files=sorted(glob.glob(os.path.join(a.data_dir,f"{pre}*.npy")))
    np.random.default_rng(0).shuffle(files); files=files[:a.n_files]
    tot=torch.zeros(4096,dtype=torch.float64,device=dev); n_ts=0
    with torch.no_grad():
        for f in files:
            arr=np.load(f); arr=arr[:,0,:] if arr.ndim==3 else arr
            arr=(2.0*(np.asarray(arr,dtype=np.float32)-fmin)/frng-1.0)
            for i in range(0,arr.shape[0],a.batch_size):
                x=torch.from_numpy(arr[i:i+a.batch_size].copy()).to(dev)
                tot+=(m.get_acts(x,indices=None,normalize=True)>0).sum(0).double()
            n_ts+=1
    mean=(tot/max(n_ts,1)).cpu().numpy(); total=tot.cpu().numpy()
    print(f"\nLayer {a.layer} | {n_ts} timesteps | per-sample Matryoshka")
    print(f"{'Group':10}{'Concepts':12}{'Mean nodes/ts':16}{'Dead':6}")
    for lo,hi,name in GROUPS:
        print(f"{name:10}{f'{lo}-{hi-1}':12}{mean[lo:hi].mean():<16.1f}{int((total[lo:hi]==0).sum()):<6d}")
if __name__=="__main__": main()
