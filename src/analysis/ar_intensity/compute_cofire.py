"""Per-node latent co-firing matrices for plain_L8 & matry_L8 (parent/child hierarchy)."""
import os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import index_to_datetime
OUT="results/ar_intensity/cofire"; N_SAMPLE=300; TOTAL=56700
def main():
    os.makedirs(OUT,exist_ok=True); dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    cand=np.random.default_rng(0).choice(TOTAL,N_SAMPLE*2,replace=False)+1
    models={n:load_sae(n,dev) for n in ["plain_L8","matry_L8"]}; c0=models["plain_L8"][1]
    acc={n:{"cofire":torch.zeros(4096,4096,device=dev,dtype=torch.float32),
            "fire":torch.zeros(4096,device=dev,dtype=torch.float32),"nodes":0} for n in models}
    used=0
    for ti in cand:
        if used>=N_SAMPLE: break
        try: a=np.load(act_file(c0,index_to_datetime(int(ti))),mmap_mode="r")
        except Exception: continue
        xr=np.ascontiguousarray(a).astype(np.float32); xr=xr.reshape(xr.shape[0],-1)
        for n,(m,c,fmin,frng) in models.items():
            x=xr if fmin is None else (2.0*(xr-fmin)/frng-1.0).astype(np.float32)
            with torch.no_grad():
                B=(encode(m,c["arch"],torch.from_numpy(x).to(dev))>0).float()
                acc[n]["cofire"]+=B.T@B; acc[n]["fire"]+=B.sum(0); acc[n]["nodes"]+=B.shape[0]
            del B
        used+=1
        if used%50==0: print(used,"/",N_SAMPLE,flush=True)
    for n in models:
        np.savez(f"{OUT}/cofire_{n}.npz",cofire=acc[n]["cofire"].cpu().numpy(),
                 fire=acc[n]["fire"].cpu().numpy(),nodes=np.array([acc[n]["nodes"]]))
        print(n,"saved, nodes",acc[n]["nodes"],flush=True)
    print("COFIRE DONE")
if __name__=="__main__": main()
