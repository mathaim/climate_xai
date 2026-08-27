"""
Per-node AR-classifier dataset builder.

For each of the 4 regions, encode the region's mesh-node SAE latents over a random sample of
timesteps and attach the per-node Higgins AR label (class_masks==2, via the index-aligned
ar_region_masks.npz). Node->cell mapping (li/ji) is the one validated in part1_align_test.py.
Self-validates on the first batch: mean IVT at AR-labeled vs non-AR nodes (AR must be >> higher).

Saves pernode_{SAE}.npz with, per region r:  {r}_X (n,4096) f16, {r}_y (n,) u8, {r}_ti (n,) i32.
Env: SAE, PER_CLASS (cap/class/region), PERTS (cap/class/region/timestep), MAXTS. SLURM GPU.
"""
import os, glob, datetime as DT
import numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.regions import REGIONS, AR_START
from src.analysis.ar_intensity.ivt_pipeline import (
    region_node_setup, load_channel_index, node_ivt, NPZ, ERA5_DIR)

SAE       = os.environ.get("SAE", "plain_L8")
PER_CLASS = int(os.environ.get("PER_CLASS", "80000"))
PERTS     = int(os.environ.get("PERTS", "150"))
MAXTS     = int(os.environ.get("MAXTS", "14000"))
OUT       = os.environ.get("OUT", f"/scratch/euh7ys/climate_xai/concept_ivt/pernode_{SAE}.npz")

def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("SAE",SAE,"dev",dev, flush=True)
    setup = region_node_setup(); d = np.load(NPZ)
    masks = {r: d[f"{r}__mask"] for r in REGIONS}; Tm = {r: masks[r].shape[0] for r in REGIONS}
    unodes = np.unique(np.concatenate([setup[r]["nodes"] for r in REGIONS]))
    posmap = {int(x): i for i, x in enumerate(unodes)}
    ridx = {r: np.array([posmap[int(x)] for x in setup[r]["nodes"]]) for r in REGIONS}
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m, c, fmin, frng = load_sae(SAE, dev)

    files = sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
    rng = np.random.default_rng(0); rng.shuffle(files); files = files[:MAXTS]
    print(f"scan up to {len(files)} ts; region nodes:", {r: len(setup[r]['nodes']) for r in REGIONS}, flush=True)

    buf = {r: {0: [], 1: []} for r in REGIONS}; cnt = {r: {0: 0, 1: 0} for r in REGIONS}
    tib = {r: {0: [], 1: []} for r in REGIONS}
    full = lambda r,cl: cnt[r][cl] >= PER_CLASS
    checked = False
    for k, f in enumerate(files):
        if all(full(r,cl) for r in REGIONS for cl in (0,1)): break
        dt = pdt(os.path.basename(f)); ti = int(round((dt-AR_START).total_seconds()/21600)) + 1
        if any(ti-1 < 0 or ti-1 >= Tm[r] for r in REGIONS): continue
        a = np.load(f, mmap_mode="r")
        xr = np.ascontiguousarray(a[unodes]).astype(np.float32).reshape(len(unodes), -1)
        x = xr if fmin is None else (2.0*(xr-fmin)/frng - 1.0).astype(np.float32)
        with torch.no_grad():
            acts = encode(m, c["arch"], torch.from_numpy(x).to(dev)).clamp_min(0).cpu().numpy().astype(np.float16)
        if not checked:
            era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy", mmap_mode="r")
            iv_all = np.asarray(node_ivt(np.ascontiguousarray(era[unodes]), qi, ui, vi, levels))
        for r in REGIONS:
            lab = (masks[r][ti-1][setup[r]["li"], setup[r]["ji"]] > 0).astype(np.uint8)
            ar_c = acts[ridx[r]]
            if not checked and lab.any() and (lab==0).any():
                ivr = iv_all[ridx[r]]
                print(f"  [CHECK {r}] ti={ti} IVT@AR={ivr[lab==1].mean():.0f} "
                      f"IVT@noAR={ivr[lab==0].mean():.0f}  nAR={int(lab.sum())}/{len(lab)}", flush=True)
            for cl in (0,1):
                if full(r,cl): continue
                sel = np.where(lab==cl)[0]
                if len(sel)==0: continue
                take = min(len(sel), PERTS, PER_CLASS - cnt[r][cl])
                sel = sel[:take]; buf[r][cl].append(ar_c[sel]); cnt[r][cl]+=take
                tib[r][cl].append(np.full(take, ti, np.int32))
        checked = True
        if (k+1) % 500 == 0:
            print(f"  {k+1} ts | " + " ".join(f"{r}:{cnt[r][1]}/{cnt[r][0]}" for r in REGIONS), flush=True)

    save = {}
    for r in REGIONS:
        if not (buf[r][1] and buf[r][0]): print(f"WARN {r}: empty class", flush=True); continue
        save[f"{r}_X"]  = np.concatenate([np.concatenate(buf[r][1]), np.concatenate(buf[r][0])])
        save[f"{r}_y"]  = np.concatenate([np.ones(cnt[r][1],np.uint8), np.zeros(cnt[r][0],np.uint8)])
        save[f"{r}_ti"] = np.concatenate([np.concatenate(tib[r][1]), np.concatenate(tib[r][0])])
        print(f"{r}: AR={cnt[r][1]} nonAR={cnt[r][0]}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); np.savez(OUT, **save); print("saved", OUT, flush=True)

if __name__ == "__main__": main()
