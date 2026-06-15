#!/usr/bin/env python3
"""Stream a layer's activations, accumulate per-feature global min/max,
early-stop when the extremes stop moving, save feature_min/max.npy into the
data dir (what the Matryoshka loader expects for [-1,1] normalization)."""
import argparse, glob, os
import numpy as np
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    a=ap.parse_args()
    pre=f"layer{a.layer:04d}_mesh_gnn_post_res"
    files=sorted(glob.glob(os.path.join(a.data_dir,f"{pre}*.npy")))
    assert files, f"no files in {a.data_dir}"
    np.random.default_rng(0).shuffle(files)
    print(f"{len(files)} files; streaming min/max", flush=True)
    fmin=fmax=prev=None; stable=n=0
    for f in files:
        arr=np.load(f,mmap_mode="r"); arr=arr[:,0,:] if arr.ndim==3 else arr
        arr=np.asarray(arr,dtype=np.float32); mn=arr.min(0); mx=arr.max(0)
        fmin=mn if fmin is None else np.minimum(fmin,mn)
        fmax=mx if fmax is None else np.maximum(fmax,mx); n+=1
        if n%a.check_every==0:
            cur=np.concatenate([fmin,fmax])
            if prev is not None:
                ch=float(np.max(np.abs(cur-prev))); print(f"  {n} files | max change={ch:.4g}",flush=True)
                stable=stable+1 if ch<a.tol else 0
                if stable>=a.patience: print(f"  converged after {n} files",flush=True); break
            prev=cur.copy()
    np.save(os.path.join(a.data_dir,"feature_min.npy"),fmin.astype(np.float32))
    np.save(os.path.join(a.data_dir,"feature_max.npy"),fmax.astype(np.float32))
    r=fmax-fmin; print(f"saved; range min={r.min():.4g} max={r.max():.4g} median={np.median(r):.4g}",flush=True)
if __name__=="__main__": main()
