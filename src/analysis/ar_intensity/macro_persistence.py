"""Full-dictionary cross-layer persistence: firing co-occurrence of ALL 4096 L8 concepts vs ALL
4096 L15 latents over NMAX stratified timesteps. Saves counts npz; plotting is offline."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
NMAX = int(os.environ.get("NMAX", "8000")); THRESH = 0.1
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/macro_persistence.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("REQUIRE_GPU") == "1": assert torch.cuda.is_available()
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""),"%Y-%m-%dT%H-%M")
def enc(f, m, c, fmin, frng):
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x).to(DEV))
def main():
    print("device", DEV, "NMAX", NMAX, flush=True)
    m8,c8,f8m,f8r = load_sae("matry_L8", DEV); m15,c15,f15m,f15r = load_sae("matry_L15", DEV)
    dtmap = lambda dd: {pdt(os.path.basename(f)): f for f in glob.glob(f"{dd}/layer*_*.npy")}
    F8,F15 = dtmap(c8["act"]), dtmap(c15["act"]); shared = sorted(set(F8)&set(F15))
    sel = shared if NMAX >= len(shared) else [shared[i] for i in np.linspace(0,len(shared)-1,NMAX).astype(int)]
    print("shared", len(shared), "using", len(sel), flush=True)
    both = torch.zeros(4096,4096, dtype=torch.float64, device=DEV)
    cnt8 = torch.zeros(4096, dtype=torch.float64, device=DEV); cnt15 = torch.zeros(4096, dtype=torch.float64, device=DEV)
    for i,dt in enumerate(sel):
        with torch.no_grad():
            B8 = (enc(F8[dt],m8,c8,f8m,f8r) > THRESH).float(); B15 = (enc(F15[dt],m15,c15,f15m,f15r) > THRESH).float()
            both += (B8.T@B15).double(); cnt8 += B8.sum(0).double(); cnt15 += B15.sum(0).double()
        if (i+1)%1000==0 or (i+1)==len(sel):
            np.savez(OUT, both=both.cpu().numpy(), cnt8=cnt8.cpu().numpy(), cnt15=cnt15.cpu().numpy(), nsteps=i+1)
            print(f"  {i+1}/{len(sel)} (checkpointed)", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
