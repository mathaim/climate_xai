#!/usr/bin/env python3
"""Count dead latents in a trained SAE. dead = latent that never activates.
For Matryoshka, applies the same [-1,1] feature normalization the trainer used
(feature_min/max in the data dir) so the eval is in-distribution."""
import argparse, glob, os
import numpy as np
import torch
from src.models.plain_sae import PlainSAE, topk
from src.models.matryoshka_sae import MatryoshkaSAE

def list_files(data_dir, layer):
    pre=f"layer{layer:04d}_mesh_gnn_post_res"
    fs=sorted(glob.glob(os.path.join(data_dir,f"{pre}*.npy")))
    if not fs:
        fs=sorted(f for f in glob.glob(os.path.join(data_dir,"*.npy"))
                  if not any(s in f for s in ["feature_min","feature_max","feature_std"]))
    return fs

def batches(files, bs, d_in, n_sample):
    seen=0
    for f in files:
        if seen>=n_sample: return
        try: arr=np.load(f,mmap_mode="r")
        except Exception as e: print(f"[skip] {os.path.basename(f)}: {e}"); continue
        if arr.ndim==3 and arr.shape[1]==1: arr=arr[:,0,:]
        if arr.ndim!=2 or arr.shape[1]!=d_in:
            print(f"[skip] {os.path.basename(f)}: shape {arr.shape}"); continue
        arr=np.asarray(arr,dtype=np.float32)
        for i in range(0,arr.shape[0],bs):
            yield arr[i:i+bs]; seen+=min(bs,arr.shape[0]-i)
            if seen>=n_sample: return

def plain_code(m,x):
    x=x-x.mean(1,keepdim=True); x=x/x.norm(dim=1,keepdim=True).clamp_min(1e-6)
    return topk(torch.relu(m.enc(x-m.b_pre)), m.k_active)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch",required=True,choices=["plain","matryoshka"])
    ap.add_argument("--layer",type=int,required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--data_dir",required=True)
    ap.add_argument("--n_latents",type=int,default=4096)
    ap.add_argument("--d_in",type=int,default=512)
    ap.add_argument("--n_sample",type=int,default=5_000_000)
    ap.add_argument("--batch_size",type=int,default=4096)
    ap.add_argument("--topk_mode",default="batch",choices=["batch","per_sample"])
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(a.checkpoint,map_location=dev)
    state=ck["model_state_dict"] if isinstance(ck,dict) and "model_state_dict" in ck else ck
    if a.arch=="plain":
        m=PlainSAE(d_in=a.d_in,n_latents=a.n_latents)
    else:
        m=MatryoshkaSAE(d_model=a.d_in,n_latents=a.n_latents,
                        group_sizes=[256,512,1024,2048,4096],target_l0=32,n_steps=300000,topk_mode=a.topk_mode)
    miss=m.load_state_dict(state,strict=False)
    if miss.missing_keys: print("  missing keys:",miss.missing_keys)
    m.to(dev).eval()
    # match Matryoshka training: [-1,1] feature normalization if stats exist
    fmin=frng=None
    if a.arch=="matryoshka":
        mp=os.path.join(a.data_dir,"feature_min.npy"); xp=os.path.join(a.data_dir,"feature_max.npy")
        if os.path.exists(mp) and os.path.exists(xp):
            fmin=np.load(mp).astype(np.float32); fmax=np.load(xp).astype(np.float32)
            frng=fmax-fmin; frng[frng<1e-8]=1.0
            print("  applying [-1,1] feature normalization (matches training)")
        else:
            print("  no feature_min/max in data dir -> raw input")
    files=list_files(a.data_dir,a.layer)
    print(f"[{a.arch} L{a.layer}] {len(files)} files; target {a.n_sample:,} vectors",flush=True)
    fire=torch.zeros(a.n_latents,dtype=torch.long,device=dev); total=0
    with torch.no_grad():
        for ch in batches(files,a.batch_size,a.d_in,a.n_sample):
            if fmin is not None: ch=(2.0*(ch-fmin)/frng-1.0).astype(np.float32)
            x=torch.from_numpy(np.ascontiguousarray(ch)).to(dev)
            code=plain_code(m,x) if a.arch=="plain" else m.get_acts(x,indices=None,normalize=True)
            fire+=(code>0).sum(0).long(); total+=x.shape[0]
    freq=fire.float()/max(total,1)
    dead=int((fire==0).sum()); lt6=int((freq<1e-6).sum()); lt5=int((freq<1e-5).sum())
    print(f"RESULT arch={a.arch} layer={a.layer} n_latents={a.n_latents} samples={total:,} "
          f"| dead(0-fire)={dead}  near-dead(<1e-6)={lt6}  (<1e-5)={lt5}  "
          f"alive%={100*(a.n_latents-dead)/a.n_latents:.2f}",flush=True)

if __name__=="__main__": main()
