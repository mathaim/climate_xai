"""Uniform cross-layer tracking (full Balcells suite), on-device, with periodic checkpointing so a
partial run is recoverable. Counterpart = argmax firing-Jaccard; reports Pearson/Jaccard/Suff/Nec.
Reads only existing L8/L15 activations (no ERA5)."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
NMAX = int(os.environ.get("NMAX", "8000")); THRESH = 0.1; CKPT = 5000
CKF = "/scratch/euh7ys/climate_xai/concept_ivt/xlmet_ckpt.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("REQUIRE_GPU") == "1":
    assert torch.cuda.is_available(), "REQUIRE_GPU=1 but no CUDA (need GPU node + CUDA torch build)"
CONCEPTS = [(340,"geo-parent"),(3481,"geo-child"),(3948,"geo-child"),(3675,"geo-child"),
            (99,"AR-core"),(1454,"AR-child"),(3392,"AR-child"),(2722,"AR-child"),
            (176,"gen-AR"),(369,"gen-AR"),(664,"gen-AR"),
            (512,"gen-nest-p"),(1308,"gen-nest-c"),(230,"gen-nest-p"),(4094,"gen-nest-c")]
CC = [c[0] for c in CONCEPTS]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""),"%Y-%m-%dT%H-%M")
def enc(f, m, c, fmin, frng):
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
    if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x).to(DEV))
def report(ns, acc, n):
    b, c8, c15, xy, X_, Y_, xx, yy = [t.cpu().numpy() for t in acc]
    np.savez(CKF, nsteps=ns, both=b, cnt8=c8, cnt15=c15, sxy=xy, sx=X_, sy=Y_, sxx=xx, syy=yy, n=n, CC=np.array(CC))
    print(f"\n=== after {ns} timesteps ===\n{'concept':>8}{'kind':>12}{'L15':>6}{'Pear':>7}{'Jac':>7}{'Suff':>7}{'Nec':>7}", flush=True)
    for i,(cc,kind) in enumerate(CONCEPTS):
        J = b[i]/np.maximum(c8[i]+c15-b[i],1); j = int(np.argmax(J))
        suff = b[i,j]/max(c8[i],1); nec = b[i,j]/max(c15[j],1)
        num = n*xy[i,j]-X_[i]*Y_[j]; den = np.sqrt(max(n*xx[i]-X_[i]**2,1e-9)*max(n*yy[j]-Y_[j]**2,1e-9))
        print(f"{cc:>8}{kind:>12}{j:>6}{num/den:>7.3f}{J[j]:>7.3f}{suff:>7.3f}{nec:>7.3f}", flush=True)
def main():
    print("device", DEV, "NMAX", NMAX, flush=True)
    m8,c8,fmin8,frng8 = load_sae("matry_L8", DEV); m15,c15,fmin15,frng15 = load_sae("matry_L15", DEV)
    dtmap = lambda dd: {pdt(os.path.basename(f)): f for f in glob.glob(f"{dd}/layer*_*.npy")}
    f8,f15 = dtmap(c8["act"]), dtmap(c15["act"]); shared = sorted(set(f8) & set(f15))
    sel = shared if NMAX >= len(shared) else [shared[i] for i in np.linspace(0, len(shared)-1, NMAX).astype(int)]
    print("shared", len(shared), "using", len(sel), flush=True)
    P = len(CC); CCt = torch.tensor(CC, device=DEV); n = 0
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=DEV)
    both=z(P,4096); cnt8=z(P); cnt15=z(4096); sxy=z(P,4096); sx=z(P); sy=z(4096); sxx=z(P); syy=z(4096)
    for i,dt in enumerate(sel):
        with torch.no_grad():
            a8 = enc(f8[dt],m8,c8,fmin8,frng8); a15 = enc(f15[dt],m15,c15,fmin15,frng15)
            X = a8[:,CCt]; B8 = (X>THRESH).float(); B15 = (a15>THRESH).float()
            both += (B8.T@B15).double(); cnt8 += B8.sum(0).double(); cnt15 += B15.sum(0).double()
            sxy += (X.T@a15).double(); sx += X.sum(0).double(); sy += a15.sum(0).double()
            sxx += (X*X).sum(0).double(); syy += (a15*a15).sum(0).double(); n += X.shape[0]
        if (i+1) % CKPT == 0 or (i+1) == len(sel):
            report(i+1, (both,cnt8,cnt15,sxy,sx,sy,sxx,syy), n)
if __name__ == "__main__":
    main()
