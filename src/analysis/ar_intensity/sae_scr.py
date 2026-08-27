"""Spurious Correlation Removal (SAEBench SCR) ported to GraphCast SAEs.
Target T: node AR-strength (IVT >= 250). Spurious S: local storm season (SH & JJA, or
NH & DJF); rows restricted to extratropics (|lat| >= 20) and solstice seasons (DJF, JJA).
Biased probe trains only on T==S cells; skyline probe trains cell-balanced. Ablating the
top-n season-encoding latents from the activation should de-bias the biased probe:
SCR_n = (BA_ablated - BA_biased) / (BA_skyline - BA_biased), BA = mean per-cell accuracy
on held-out data. Random-latent ablation is the floor. Both architectures per layer."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
NMAX = int(os.environ.get("NMAX", "8000"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("REQUIRE_GPU") == "1": assert torch.cuda.is_available()
IVT_T = 250.0; NS = (5, 10, 20, 50); PER_CELL = 40
LAYERS = [("L0", "matry_L0", "plain_L0"), ("L8", "matry_L8", "plain_L8"), ("L15", "matry_L15", "plain_L15")]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")
def prep(x_raw, arch, m, fmin, frng):
    if arch == "matry":
        xs = torch.from_numpy((2.0*(x_raw - fmin)/frng - 1.0).astype(np.float32)).to(DEV)
        xn = m.normalizer.normalize(xs)
        code = m._apply_topk(xn @ m.W_enc + m.b_enc, m.target_l0)
        return xn, code, m.W_dec
    xt = torch.from_numpy(x_raw).to(DEV); xm = xt - xt.mean(1, keepdim=True)
    xn = xm / xm.norm(dim=1, keepdim=True).clamp_min(1e-6)
    from src.models.plain_sae import topk
    code = topk(torch.relu(m.enc(xn - m.b_pre)), m.k_active)
    W = m.dec.weight
    return xn, code, (W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)).T
def fit_logistic(X, y, l1=0.0, iters=400, lr=0.05):
    w = torch.zeros(X.shape[1], device=DEV, requires_grad=True); b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
        if l1 > 0: loss = loss + l1*w.abs().mean()
        loss.backward(); opt.step()
    return w.detach(), b.detach()
def bal_acc(logits, T, S):
    accs = []
    for t in (0, 1):
        for s in (0, 1):
            m = (T == t) & (S == s)
            if m.sum() > 0: accs.append((((logits[m] > 0).float() == T[m]).float().mean()).item())
    return float(np.mean(accs))
def main():
    print("device", DEV, "NMAX", NMAX, flush=True)
    SAVE = {}
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat = era0[:, lat_i].astype(float)
    extra = np.abs(nlat) >= 20.0; rng = np.random.default_rng(0)
    for L, mname, pname in LAYERS:
        mm, mc, fmin, frng = load_sae(mname, DEV); pm, pc, _, _ = load_sae(pname, DEV)
        files = sorted(glob.glob(f"{pc['act']}/layer*_*.npy"))
        sel = [f for f in [files[i] for i in np.linspace(0, len(files)-1, min(NMAX, len(files))).astype(int)]
               if pdt(os.path.basename(f)).month in (12, 1, 2, 6, 7, 8)]
        cut = len(sel)//2
        print(f"{L}: {len(sel)} solstice-season timesteps ({cut} train)", flush=True)
        def labels(f):
            dt = pdt(os.path.basename(f))
            era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
            T = node_ivt(era, qi, ui, vi, levels) >= IVT_T
            jja = dt.month in (6, 7, 8)
            S = np.where(nlat <= -20, jja, ~jja) & extra                     # local winter
            return T, S.astype(bool)
        # ---- phase A: training rows per cell ----
        store = {a: [] for a in ("matry", "plain")}; TT, SS = [], []
        for k, f in enumerate(sel[:cut]):
            try: T, S = labels(f)
            except FileNotFoundError: continue
            keep = []
            for t in (0, 1):
                for s in (0, 1):
                    cell = np.flatnonzero(extra & (T == t) & (S == s))
                    if len(cell): keep.append(rng.choice(cell, min(PER_CELL, len(cell)), replace=False))
            keep = np.concatenate(keep)
            x = np.ascontiguousarray(np.load(f, mmap_mode="r")).astype(np.float32).reshape(-1, 512)[keep]
            with torch.no_grad():
                for arch, mod in (("matry", mm), ("plain", pm)):
                    xn, code, _ = prep(x, arch, mod, fmin, frng)
                    store[arch].append((xn.cpu(), code.cpu()))
            TT.append(torch.tensor(T[keep], dtype=torch.float32)); SS.append(torch.tensor(S[keep], dtype=torch.float32))
            if (k+1) % 1000 == 0: print(f"  {L} train {k+1}/{cut}", flush=True)
        T = torch.cat(TT).to(DEV); S = torch.cat(SS).to(DEV)
        conf = (T == S)                                                       # biased training subset
        probes = {}
        for arch in ("matry", "plain"):
            X = torch.cat([a for a, _ in store[arch]]).to(DEV); C = torch.cat([c for _, c in store[arch]]).to(DEV)
            wb, bb = fit_logistic(X[conf], T[conf])                           # biased probe
            wk, bk = fit_logistic(X, T)                                       # skyline probe
            sw, _ = fit_logistic(C, S, l1=0.01)                               # season latents via L1 probe
            season_rank = torch.argsort(sw.abs(), descending=True)
            print(f"{arch}_{L}: season latents top10 {season_rank[:10].tolist()}", flush=True)
            probes[arch] = (wb, bb, wk, bk, season_rank)
            del X, C
        # ---- phase B: held-out evaluation ----
        agg = {a: {"biased": [], "sky": [], "abl": {n: [] for n in NS}, "rand": {n: [] for n in NS}} for a in ("matry", "plain")}
        for k, f in enumerate(sel[cut:]):
            try: T1, S1 = labels(f)
            except FileNotFoundError: continue
            keep = []
            for t in (0, 1):
                for s in (0, 1):
                    cell = np.flatnonzero(extra & (T1 == t) & (S1 == s))
                    if len(cell): keep.append(rng.choice(cell, min(PER_CELL, len(cell)), replace=False))
            keep = np.concatenate(keep)
            x = np.ascontiguousarray(np.load(f, mmap_mode="r")).astype(np.float32).reshape(-1, 512)[keep]
            Tv = torch.tensor(T1[keep], dtype=torch.float32, device=DEV); Sv = torch.tensor(S1[keep], dtype=torch.float32, device=DEV)
            with torch.no_grad():
                for arch, mod in (("matry", mm), ("plain", pm)):
                    wb, bb, wk, bk, rank = probes[arch]
                    xn, code, Wd = prep(x, arch, mod, fmin, frng)
                    agg[arch]["biased"].append(bal_acc(xn @ wb + bb, Tv, Sv))
                    agg[arch]["sky"].append(bal_acc(xn @ wk + bk, Tv, Sv))
                    for n in NS:
                        cols = rank[:n]
                        xa = xn - code[:, cols] @ Wd[cols]
                        agg[arch]["abl"][n].append(bal_acc(xa @ wb + bb, Tv, Sv))
                        rc = torch.from_numpy(rng.choice(4096, n, replace=False)).to(DEV)
                        xr = xn - code[:, rc] @ Wd[rc]
                        agg[arch]["rand"][n].append(bal_acc(xr @ wb + bb, Tv, Sv))
            if (k+1) % 1000 == 0: print(f"  {L} eval {k+1}/{len(sel)-cut}", flush=True)
        for arch in ("matry", "plain"):
            g = agg[arch]; B, K = np.mean(g["biased"]), np.mean(g["sky"])
            line = "  ".join(f"n={n}: {(np.mean(g['abl'][n])-B)/max(K-B,1e-9):+.3f}"
                             f"(rand {(np.mean(g['rand'][n])-B)/max(K-B,1e-9):+.3f})" for n in NS)
            print(f"SCR {arch}_{L}: biased BA {B:.3f}  skyline BA {K:.3f} | {line}", flush=True)
        for arch in ("matry", "plain"):
            g = agg[arch]
            SAVE[f"{L}_{arch}_biased"] = np.array(g["biased"]); SAVE[f"{L}_{arch}_sky"] = np.array(g["sky"])
            for n in NS:
                SAVE[f"{L}_{arch}_abl{n}"] = np.array(g["abl"][n]); SAVE[f"{L}_{arch}_rand{n}"] = np.array(g["rand"][n])
        np.savez("/scratch/euh7ys/climate_xai/patching/scr_percell.npz", **SAVE)
        print(f"saved scr_percell.npz through {L}", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
