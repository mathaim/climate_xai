"""Task 1 analysis: containment (P, lift vs independence null, bootstrap CI) of all concepts over the 3
children; AR classification (presence/intensity/non-AR) from region IVT-tuning; candidate parents."""
import numpy as np
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/task1_foundation.npz")
cofire, childfire, rate_ts = D["cofire_ts"], D["childfire_ts"], D["rate_ts"]
actsum, regmass = D["actsum"], D["regmass"]; tun_act, tun_fire, tun_cnt = D["tun_act"], D["tun_fire"], D["tun_cnt"]
bins = D["bins"]; CH = [int(x) for x in D["children"]]; nnode = int(D["nnode"]); used = int(D["used"])
ctr = (bins[:-1] + bins[1:]) / 2; Nt = cofire.shape[0]
Pmat = cofire.sum(0) / np.maximum(childfire.sum(0)[:, None], 1)      # P(concept|child_i) [nc,4096]
rate = rate_ts.sum(0) / (used * nnode)                               # per-node firing prob [4096]
regfrac = regmass / np.maximum(actsum, 1e-9)
tun_r = tun_fire / np.maximum(tun_cnt[None, :], 1)                   # firing rate per IVT bin (region)
tun_a = tun_act / np.maximum(tun_cnt[None, :], 1)
com_ivt = (tun_a * ctr[None, :]).sum(1) / np.maximum(tun_a.sum(1), 1e-9)
AR = 250.0  # standard atmospheric-river IVT threshold (physical, not tuned)
fire_dry = tun_r[:, ctr < AR].mean(1); fire_peak = tun_r.max(1) + 1e-9
# IVT at which firing rate reaches half its max (rate-based turn-on)
def ivt50(c):
    r = tun_r[c]; h = 0.5 * r.max()
    idx = np.where(r >= h)[0]; return ctr[idx[0]] if len(idx) else np.nan
def klass(c):
    if fire_peak[c] < 1e-3: return "silent"
    if fire_dry[c] > 0.4 * fire_peak[c]: return "non-AR"        # fires when dry (<250) => not AR
    return "AR-intensity" if ivt50(c) >= 600 else "AR-presence"
def bootP_min(c, B=1000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, Nt, (B, Nt)); mins = np.ones(B)
    for ci in range(len(CH)):
        num = cofire[:, ci, c][idx].sum(1); den = childfire[:, ci][idx].sum(1); mins = np.minimum(mins, num / np.maximum(den, 1))
    return np.percentile(mins, [2.5, 50, 97.5])
def curve(c): return " ".join(f"{tun_r[c, b]:.2f}" for b in range(len(ctr)))
print("IVT bin centers:", [int(x) for x in ctr]); print(f"AR threshold {AR:.0f}\n")
print("REFERENCE (children + storm-track 340):")
for c in CH + [340]:
    print(f"  {c:>5} {klass(c):>12} rate {rate[c]:.4f} regfrac {regfrac[c]:.2f} com_ivt {com_ivt[c]:.0f} ivt50 {ivt50(c):.0f} fire_dry/peak {fire_dry[c]/fire_peak[c]:.2f}")
    print(f"        firerate vs IVT: {curve(c)}")
Pmin = Pmat.min(0); liftmin = (Pmat / np.maximum(rate[None, :], 1e-9)).min(0); crate = rate[CH]
cand = [c for c in np.argsort(-liftmin) if c not in CH and rate[c] > 1.5 * crate.max() and Pmin[c] > 0.3][:20]
print(f"\nCANDIDATE PARENTS (contain all 3, >1.5x broader), ranked by min lift:")
print(f"{'con':>5}{'Pmin':>6}{'CIlo':>6}{'liftmin':>8}{'rate':>7}{'regfrac':>8}{'com_ivt':>8}{'ivt50':>7}{'class':>13}")
for c in cand:
    ci = bootP_min(c)
    print(f"{c:>5}{Pmin[c]:>6.2f}{ci[0]:>6.2f}{liftmin[c]:>8.1f}{rate[c]:>7.4f}{regfrac[c]:>8.2f}{com_ivt[c]:>8.0f}{ivt50(c):>7.0f}{klass(c):>13}")
    if klass(c) in ("AR-presence", "AR-intensity"): print(f"        firerate vs IVT: {curve(c)}")
