"""Firing-mass composition by IVT band (sub-AR <250, AR 250-500, strong-AR >500) for each
(sae, layer). Mass-conserving under feature splitting. Sequential over layer combos, checkpoints
after each. NMAX stratified timesteps per combo."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode, SAES
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
NMAX = int(os.environ.get("NMAX", "8000")); THRESH = 0.1
COMBOS = os.environ.get("COMBOS", "matry_L0,matry_L8,matry_L15,plain_L0,plain_L8,plain_L15").split(",")
OUT = "/scratch/euh7ys/climate_xai/concept_ivt/macro_ivt_mass.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = [0.0, 250.0, 500.0, 1e9]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""),"%Y-%m-%dT%H-%M")
def main():
    print("device", DEV, "NMAX", NMAX, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    res = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}
    for name in COMBOS:
        if name not in SAES: print("skip (unknown sae)", name, flush=True); continue
        if f"{name}_mass" in res: print("skip (done)", name, flush=True); continue
        try: m, c, fmin, frng = load_sae(name, DEV)
        except Exception as e: print(f"skip {name}: {e}", flush=True); continue
        files = sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
        if not files: print("skip (no acts)", name, flush=True); continue
        sel = files if NMAX >= len(files) else [files[i] for i in np.linspace(0,len(files)-1,NMAX).astype(int)]
        mass = np.zeros(3); tot = np.zeros(3); used = 0
        for k,f in enumerate(sel):
            dt = pdt(os.path.basename(f))
            try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
            except FileNotFoundError: continue
            iv = node_ivt(era, qi, ui, vi, levels); bi = np.clip(np.digitize(iv, BANDS)-1, 0, 2)
            a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
            if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
            with torch.no_grad(): nf = (encode(m, c["arch"], torch.from_numpy(x).to(DEV)) > THRESH).sum(1).cpu().numpy()
            for b in range(3): mass[b] += nf[bi==b].sum(); tot[b] += (bi==b).sum()
            used += 1
            if (k+1)%1000==0: print(f"  {name} {k+1}/{len(sel)}", flush=True)
        res[f"{name}_mass"] = mass; res[f"{name}_tot"] = tot; res[f"{name}_n"] = used
        np.savez(OUT, **res)
        fr = mass/max(mass.sum(),1)
        print(f"{name}: n={used}  mass sub-AR {fr[0]:.3f}  AR {fr[1]:.3f}  strong {fr[2]:.3f}  (checkpointed)", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
