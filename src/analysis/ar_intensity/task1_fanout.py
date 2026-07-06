"""Children of the AR parent 1829: concepts contained in it (P(1829|C) high, narrower), with AR class.
Uses the existing foundation encode (1829's co-firing with all concepts is already recorded)."""
import os, numpy as np
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/task1_foundation.npz")
cofire, rate_ts = D["cofire_ts"], D["rate_ts"]; tun_fire, tun_cnt = D["tun_fire"], D["tun_cnt"]
regmass, actsum = D["regmass"], D["actsum"]; bins = D["bins"]; CH = [int(x) for x in D["children"]]
used = int(D["used"]); nnode = int(D["nnode"]); ctr = (bins[:-1] + bins[1:]) / 2
PARENT = int(os.environ.get('PARENT', '1829')); pidx = CH.index(PARENT)
rate = rate_ts.sum(0) / (used * nnode); regfrac = regmass / np.maximum(actsum, 1e-9)
tun_r = tun_fire / np.maximum(tun_cnt[None, :], 1); regfire = tun_fire.sum(1)
com = (tun_r * ctr[None, :]).sum(1) / np.maximum(tun_r.sum(1), 1e-9)
AR = 250.0; fire_dry = tun_r[:, ctr < AR].mean(1); fire_peak = tun_r.max(1) + 1e-9
def ivt50(c):
    r = tun_r[c]; idx = np.where(r >= 0.5 * r.max())[0]; return ctr[idx[0]] if len(idx) else np.nan
def klass(c):
    if regfire[c] < 150: return "rare"
    if fire_dry[c] > 0.4 * fire_peak[c]: return "non-AR"
    return "AR-intensity" if ivt50(c) >= 600 else "AR-presence"
cof = cofire[:, pidx, :].sum(0); fireC = rate * used * nnode
Pp = cof / np.maximum(fireC, 1)   # P(1829 | C): ~1 => C contained in 1829
cand = [c for c in np.argsort(-Pp) if c != PARENT and rate[c] < rate[PARENT] and Pp[c] > 0.5][:20]
print(f"parent {PARENT} ({klass(PARENT)}, rate {rate[PARENT]:.4f}, regfrac {regfrac[PARENT]:.2f}, regfire {int(regfire[PARENT])})")
print(f"{'C':>5}{'P(1829|C)':>10}{'rate':>8}{'regfrac':>8}{'com_ivt':>8}{'ivt50':>7}{'class':>13}{'regfire':>8}")
for c in cand:
    print(f"{c:>5}{Pp[c]:>10.2f}{rate[c]:>8.4f}{regfrac[c]:>8.2f}{com[c]:>8.0f}{ivt50(c):>7.0f}{klass(c):>13}{int(regfire[c]):>8}")
