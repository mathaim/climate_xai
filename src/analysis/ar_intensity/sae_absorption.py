"""Feature absorption (Chanin et al., arXiv:2409.14507) ported to GraphCast SAEs.
Concept: node has AR-strength air (local IVT >= 250). Chronological split: first half of
8000 stratified timesteps trains probes, second half is scored. Main latents: L1-logistic
(coef 0.01) on latent codes, top-k by |coef|, k grown while F1 jumps > tau=0.03. Absorption
instance: activation-probe true positive where all main latents are silent and the
top-ablation-effect latent (cos(decoder, probe) >= 0.025, not a main) dominates the
runner-up by >= margin. Ablation effect is exact here: code_j * (W_dec[j] . w). Rates
reported per margin in {0.5, 1.0, 2.0}, normalized by probe true positives and by
main-silent true positives."""
import os, glob, numpy as np, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
NMAX = int(os.environ.get("NMAX", "8000"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("REQUIRE_GPU") == "1": assert torch.cuda.is_available()
IVT_T, COS_MIN, TAU, MARGINS = 250.0, 0.025, 0.03, (0.5, 1.0, 2.0)
LAYERS = [("L0", "matry_L0", "plain_L0"), ("L8", "matry_L8", "plain_L8"), ("L15", "matry_L15", "plain_L15")]
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")
def prep(x_raw, arch, m, fmin, frng):
    """Return (operating-space input, raw code that multiplies the decoder, effective decoder)."""
    if arch == "matry":
        xs = torch.from_numpy((2.0*(x_raw - fmin)/frng - 1.0).astype(np.float32)).to(DEV)
        xn = m.normalizer.normalize(xs)
        code = m._apply_topk(xn @ m.W_enc + m.b_enc, m.target_l0)
        return xn, code, m.W_dec                                             # (4096, 512)
    xt = torch.from_numpy(x_raw).to(DEV); xm = xt - xt.mean(1, keepdim=True)
    xn = xm / xm.norm(dim=1, keepdim=True).clamp_min(1e-6)
    from src.models.plain_sae import topk
    code = topk(torch.relu(m.enc(xn - m.b_pre)), m.k_active)
    W = m.dec.weight                                                          # (512, 4096)
    return xn, code, (W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)).T      # (4096, 512)
def fit_logistic(X, y, l1=0.0, iters=400, lr=0.05):
    w = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
        if l1 > 0: loss = loss + l1 * w.abs().mean()
        loss.backward(); opt.step()
    return w.detach(), b.detach()
def f1(logits, y):
    p = logits > 0
    tp = (p & (y > .5)).sum().item(); fp = (p & (y < .5)).sum().item(); fn = ((~p) & (y > .5)).sum().item()
    return 2*tp / max(2*tp + fp + fn, 1)
def main():
    print("device", DEV, "NMAX", NMAX, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    rng = np.random.default_rng(0)
    for L, mname, pname in LAYERS:
        mm, mc, fmin, frng = load_sae(mname, DEV); pm, pc, _, _ = load_sae(pname, DEV)
        files = sorted(glob.glob(f"{pc['act']}/layer*_*.npy"))
        sel = [files[i] for i in np.linspace(0, len(files)-1, min(NMAX, len(files))).astype(int)]
        cut = len(sel)//2
        # ---------- phase A: training subsample ----------
        rows = {"matry": {"X": [], "C": []}, "plain": {"X": [], "C": []}}; Ys = []
        for k, f in enumerate(sel[:cut]):
            dt = pdt(os.path.basename(f))
            try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
            except FileNotFoundError: continue
            y = node_ivt(era, qi, ui, vi, levels) >= IVT_T
            pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
            if len(pos) < 20: continue
            keep = np.concatenate([rng.choice(pos, min(100, len(pos)), replace=False),
                                   rng.choice(neg, 100, replace=False)])
            x = np.ascontiguousarray(np.load(f, mmap_mode="r")).astype(np.float32).reshape(-1, 512)[keep]
            with torch.no_grad():
                for arch, mod in (("matry", mm), ("plain", pm)):
                    xn, code, _ = prep(x, arch, mod, fmin, frng)
                    rows[arch]["X"].append(xn.cpu()); rows[arch]["C"].append(code.cpu())
            Ys.append(torch.tensor(y[keep], dtype=torch.float32))
            if (k+1) % 1000 == 0: print(f"  {L} train {k+1}/{cut}", flush=True)
        Y = torch.cat(Ys).to(DEV)
        probes = {}
        for arch, mod in (("matry", mm), ("plain", pm)):
            X = torch.cat(rows[arch]["X"]).to(DEV); C = torch.cat(rows[arch]["C"]).to(DEV)
            w, b = fit_logistic(X, Y)
            print(f"{arch}_{L}: activation-probe train F1 {f1(X @ w + b, Y):.3f}", flush=True)
            sub = torch.randperm(C.shape[0], device=DEV)[:100_000]
            lw, lb = fit_logistic(C[sub], Y[sub], l1=0.01)
            order = torch.argsort(lw.abs(), descending=True)
            prevF, mains = 0.0, []
            for kk in range(1, 7):
                cols = order[:kk]
                sw, sb = fit_logistic(C[sub][:, cols], Y[sub], iters=300)
                Fk = f1(C[sub][:, cols] @ sw + sb, Y[sub])
                if kk == 1 or Fk - prevF > TAU: mains = cols.tolist(); prevF = Fk
                else: break
            print(f"{arch}_{L}: main latents {mains} (F1 {prevF:.3f})", flush=True)
            probes[arch] = (w, b, mains)
            del X, C
        # ---------- phase B: scoring ----------
        stats = {a: {"tp": 0, "silent": 0, "events": {g: 0 for g in MARGINS}} for a in ("matry", "plain")}
        for k, f in enumerate(sel[cut:]):
            dt = pdt(os.path.basename(f))
            try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
            except FileNotFoundError: continue
            y = torch.tensor(node_ivt(era, qi, ui, vi, levels) >= IVT_T, device=DEV)
            x = np.ascontiguousarray(np.load(f, mmap_mode="r")).astype(np.float32).reshape(-1, 512)
            with torch.no_grad():
                for arch, mod in (("matry", mm), ("plain", pm)):
                    w, b, mains = probes[arch]
                    xn, code, Wd = prep(x, arch, mod, fmin, frng)
                    proj = Wd @ w                                             # (4096,) exact ablation weights
                    cosv = (Wd / Wd.norm(dim=1, keepdim=True).clamp_min(1e-8)) @ (w / w.norm())
                    tp = ((xn @ w + b) > 0) & y
                    silent = tp & (code[:, mains].amax(1) <= 0)
                    stats[arch]["tp"] += int(tp.sum()); stats[arch]["silent"] += int(silent.sum())
                    if silent.any():
                        eff = code[silent] * proj                             # (n, 4096)
                        eff[:, mains] = -1e9
                        eff = torch.where((cosv >= COS_MIN) & (torch.arange(4096, device=DEV) == torch.arange(4096, device=DEV)), eff, eff)  # keep shape
                        top2 = eff.topk(2, dim=1)
                        ok_cos = cosv[top2.indices[:, 0]] >= COS_MIN
                        gap = top2.values[:, 0] - top2.values[:, 1]
                        for g in MARGINS:
                            stats[arch]["events"][g] += int((ok_cos & (gap >= g) & (top2.values[:, 0] > 0)).sum())
            if (k+1) % 1000 == 0: print(f"  {L} score {k+1}/{len(sel)-cut}", flush=True)
        for arch in ("matry", "plain"):
            s = stats[arch]
            rates = "  ".join(f"m{g}: {s['events'][g]/max(s['tp'],1):.4f}" for g in MARGINS)
            print(f"ABSORPTION {arch}_{L}: TPs {s['tp']}  mains-silent {s['silent']} "
                  f"({s['silent']/max(s['tp'],1):.3f}) | rate/TP {rates} | "
                  f"rate/silent m1.0 {s['events'][1.0]/max(s['silent'],1):.3f}", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
