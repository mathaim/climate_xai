"""Structural recurrence: run the L8 containment criteria fresh on another layer's SAE.
Per-node co-firing of all 4096 concepts over NMAX stratified timesteps; report the tightest
parent(0-1023) -> child(1024-4095) nestings with peak locations. No reference to L8."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
NMAX = int(os.environ.get("NMAX", "8000")); THRESH = 0.1
SAELIST = os.environ.get("SAELIST", "matry_L0,matry_L15").split(",")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("REQUIRE_GPU") == "1": assert torch.cuda.is_available()
conv = lambda x: x - 360 if x > 180 else x
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""),"%Y-%m-%dT%H-%M")
def main():
    print("device", DEV, "NMAX", NMAX, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i].astype(float); nlon = np.array([conv(x) for x in era0[:, lon_i].astype(float)])
    for name in SAELIST:
        out = f"/scratch/euh7ys/climate_xai/concept_ivt/nesting_scan_{name}.npz"
        m, c, fmin, frng = load_sae(name, DEV)
        files = sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
        sel = files if NMAX >= len(files) else [files[i] for i in np.linspace(0,len(files)-1,NMAX).astype(int)]
        print(f"\n===== {name}: {len(sel)} timesteps =====", flush=True)
        both = torch.zeros(4096,4096, dtype=torch.float64, device=DEV)
        cnt = torch.zeros(4096, dtype=torch.float64, device=DEV)
        fnode = torch.zeros(40962,4096, dtype=torch.float32, device=DEV)
        for i,f in enumerate(sel):
            a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
            if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
            with torch.no_grad():
                B = (encode(m, c["arch"], torch.from_numpy(x).to(DEV)) > THRESH).float()
                both += (B.T@B).double(); cnt += B.sum(0).double(); fnode += B
            if (i+1)%1000==0: print(f"  {i+1}/{len(sel)}", flush=True)
        bothn, cntn, fn = both.cpu().numpy(), cnt.cpu().numpy(), fnode.cpu().numpy()
        np.savez(out, both=bothn, cnt=cntn, peak=fn.argmax(0), peakshare=fn.max(0)/np.maximum(cntn,1))
        print(f"saved {out}", flush=True)
        P = bothn[:1024, 1024:] / np.maximum(cntn[1024:], 1)[None, :]   # P(parent|child)
        rows = []
        for cj in range(3072):
            child = 1024 + cj
            if cntn[child] < 1000: continue
            pi = int(np.argmax(P[:, cj]))
            if cntn[pi] < 2*cntn[child]: continue
            rows.append((P[pi,cj], pi, child))
        rows.sort(reverse=True)
        print(f"top tight nestings ({name}):", flush=True)
        for p, par, ch in rows[:20]:
            pk_p, pk_c = fn[:,par].argmax(), fn[:,ch].argmax()
            print(f"  P={p:.2f}  {par}({int(cntn[par])})@({nlat[pk_p]:.0f},{nlon[pk_p]:.0f})"
                  f"  ->  {ch}({int(cntn[ch])})@({nlat[pk_c]:.0f},{nlon[pk_c]:.0f})", flush=True)
        chile = [(p,par,ch) for p,par,ch in rows if p>=0.8 and -60<=nlat[fn[:,ch].argmax()]<=-30 and -80<=nlon[fn[:,ch].argmax()]<=-65]
        print(f"Chilean-coast tight nestings (P>=0.8): {[(round(p,2),par,ch) for p,par,ch in chile[:10]]}", flush=True)
if __name__ == "__main__":
    main()
