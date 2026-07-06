"""For the AR-intensity set, list each concept's top containers (P, lift vs null, AR class) and the
pairwise containment among the set, to pick a broad AR parent over localized AR children."""
import numpy as np
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/task1_foundation.npz")
cofire, childfire, rate_ts = D["cofire_ts"], D["childfire_ts"], D["rate_ts"]
tun_fire, tun_cnt = D["tun_fire"], D["tun_cnt"]; bins = D["bins"]; CH = [int(x) for x in D["children"]]
nnode = int(D["nnode"]); used = int(D["used"]); ctr = (bins[:-1] + bins[1:]) / 2
Pmat = cofire.sum(0) / np.maximum(childfire.sum(0)[:, None], 1); rate = rate_ts.sum(0) / (used * nnode)
tun_r = tun_fire / np.maximum(tun_cnt[None, :], 1); regfire = tun_fire.sum(1)
AR = 250.0; fire_dry = tun_r[:, ctr < AR].mean(1); fire_peak = tun_r.max(1) + 1e-9
def ivt50(c):
    r = tun_r[c]; idx = np.where(r >= 0.5 * r.max())[0]; return ctr[idx[0]] if len(idx) else np.nan
def klass(c):
    if regfire[c] < 150: return "rare"
    if fire_dry[c] > 0.4 * fire_peak[c]: return "non-AR"
    return "AR-intensity" if ivt50(c) >= 600 else "AR-presence"
def bootP(c, ci, B=1000):
    rng = np.random.default_rng(0); idx = rng.integers(0, cofire.shape[0], (B, cofire.shape[0]))
    num = cofire[:, ci, c][idx].sum(1); den = childfire[:, ci][idx].sum(1); return np.percentile(num / np.maximum(den, 1), [2.5, 97.5])
for ci, ch in enumerate(CH):
    P = Pmat[ci]; lift = P / np.maximum(rate, 1e-9)
    cand = [c for c in np.argsort(-lift) if c != ch and rate[c] > rate[ch]][:8]
    print(f"\n=== containers of {ch} ({klass(ch)}, rate {rate[ch]:.4f}) ===")
    print(f"{'C':>5}{'P(C|ch)':>9}{'lift':>7}{'rate':>8}{'class':>13}")
    for c in cand:
        print(f"{c:>5}{P[c]:>9.2f}{lift[c]:>7.1f}{rate[c]:>8.4f}{klass(c):>13}")
print("\n=== pairwise P(row contains col) among the AR set ===")
print("        " + "".join(f"{ch:>7}" for ch in CH))
for ri, r in enumerate(CH):
    row = []
    for ci, c in enumerate(CH):
        row.append(Pmat[ci, r])  # P(r | c) = fraction of c's firings where r also fires
    print(f"{r:>7} " + "".join(f"{v:>7.2f}" for v in row))
