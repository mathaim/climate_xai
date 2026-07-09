"""SAEBench-style core metrics adapted to GraphCast SAEs (cite Karvonen et al., SAEBench):
FVU/explained variance, effective L0, dead latents, density stats over NMAX stratified
timesteps; plus k-sparse probing (regional AR intensity, time-split R2) from cached features."""
import os, glob, numpy as np, pandas as pd, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
NMAX = int(os.environ.get("NMAX", "1000")); THRESH = 0.1
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
SAELIST = os.environ.get("SAES", "matry_L0,matry_L8,matry_L15,plain_L0,plain_L8,plain_L15").split(",")
def recon_pair(m, arch, x):
    """Return (reconstruction, target) in the SAE's own operating space."""
    if hasattr(m, "normalizer") and m.normalizer is not None:
        # Matryoshka: no forward; build recon from primitives in NORMALIZED space (exact,
        # scale cancels; normalize/unnormalize are true inverses on the 512-dim input).
        xn = m.normalizer.normalize(x)
        code = m._apply_topk(xn @ m.W_enc + m.b_enc, m.target_l0)   # raw code, as in training step()
        return code @ m.W_dec + m.b_dec, xn
    out = m(x)
    if isinstance(out, (tuple, list)): out = out[0]
    if out.shape == x.shape:
        xm = x - x.mean(1, keepdim=True)
        xn = xm / xm.norm(dim=1, keepdim=True).clamp_min(1e-6)   # forward's own target space
        return out, xn
    raise RuntimeError(f"no recon interface; model attrs: {[a for a in dir(m) if not a.startswith('_')]}")
def core_metrics(name):
    m, c, fmin, frng = load_sae(name, DEV)
    files = sorted(glob.glob(f"{c['act']}/layer*_*.npy"))
    sel = [files[i] for i in np.linspace(0, len(files)-1, min(NMAX, len(files))).astype(int)]
    sse = 0.0; svar = 0.0; l0 = 0.0; n = 0; fired = torch.zeros(4096, dtype=torch.bool, device=DEV); dens = torch.zeros(4096, dtype=torch.float64, device=DEV)
    for k, f in enumerate(sel):
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
        xt = torch.from_numpy(x).to(DEV)
        with torch.no_grad():
            acts = encode(m, c["arch"], xt); xr, tgt = recon_pair(m, c["arch"], xt)
            B = acts > THRESH
            sse += float(((xr - tgt)**2).sum()); svar += float(((tgt - tgt.mean(0))**2).sum())
            l0 += float(B.sum()); n += xt.shape[0]; fired |= B.any(0); dens += B.sum(0).double()
        if (k+1) % 250 == 0: print(f"  {name} {k+1}/{len(sel)}", flush=True)
    fvu = sse/max(svar,1e-9); dead = 4096 - int(fired.sum()); d = (dens/max(n,1)).cpu().numpy()
    return dict(fvu=fvu, expl_var=1-fvu, eff_l0=l0/n, dead=dead, med_density=float(np.median(d[d>0])))
def probing(name):
    meta = pd.read_parquet(f"{D}/{name}_meta.parquet"); F = np.load(f"{D}/{name}_features_region_binary.npy", mmap_mode="r")
    iv_all = meta["max_ivt"].values.astype(float); out = {}
    for kk in (1, 16):
        r2s = []
        for reg in meta["region"].unique():
            msk = (meta["region"]==reg).values & np.isfinite(iv_all)
            ti = meta["time_index"].values[msk]; order = np.argsort(ti)
            X = np.asarray(F[msk], float)[order]; y = iv_all[msk][order]
            ntr = int(0.8*len(y)); Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
            cz = Xtr - Xtr.mean(0); yz = ytr - ytr.mean(); den = np.sqrt((cz**2).sum(0)*(yz**2).sum())
            corr = np.where(den>0, (cz*yz[:,None]).sum(0)/np.maximum(den,1e-12), 0)
            top = np.argsort(-np.abs(corr))[:kk]
            A = np.c_[Xtr[:,top], np.ones(ntr)]; w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            pred = np.c_[Xte[:,top], np.ones(len(yte))] @ w
            r2s.append(1 - ((yte-pred)**2).sum()/((yte-yte.mean())**2).sum())
        out[f"probe_r2_k{kk}"] = float(np.mean(r2s))
    return out
def main():
    print("device", DEV, "NMAX", NMAX, flush=True)
    rows = {}
    for name in SAELIST:
        try: r = {**core_metrics(name), **probing(name)}
        except Exception as e: print(f"{name} FAILED: {e}", flush=True); continue
        rows[name] = r
        print(f"{name}: " + "  ".join(f"{k}={v:.4g}" for k, v in r.items()), flush=True)
    print(f"\n{'sae':>10}{'ExplVar':>9}{'FVU':>7}{'effL0':>7}{'dead':>6}{'medDens':>9}{'R2 k=1':>8}{'R2 k=16':>9}")
    for name, r in rows.items():
        print(f"{name:>10}{r['expl_var']:>9.3f}{r['fvu']:>7.3f}{r['eff_l0']:>7.1f}{r['dead']:>6}{r['med_density']:>9.5f}{r['probe_r2_k1']:>8.3f}{r['probe_r2_k16']:>9.3f}")
if __name__ == "__main__":
    main()
