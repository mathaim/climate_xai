"""From the Task 1 foundation: classify every concept by region IVT-tuning and list the W_S_America
AR-presence (candidate parents) and AR-intensity (candidate children) concepts. No new encode."""
import numpy as np
D = np.load("/scratch/euh7ys/climate_xai/concept_ivt/task1_foundation.npz")
tun_fire, tun_cnt = D["tun_fire"], D["tun_cnt"]; actsum, regmass = D["actsum"], D["regmass"]; rate_ts = D["rate_ts"]
bins = D["bins"]; used = int(D["used"]); nnode = int(D["nnode"])
ctr = (bins[:-1] + bins[1:]) / 2
tun_r = tun_fire / np.maximum(tun_cnt[None, :], 1)
regfrac = regmass / np.maximum(actsum, 1e-9); rate = rate_ts.sum(0) / (used * nnode); regfire = tun_fire.sum(1)
AR = 250.0; fire_dry = tun_r[:, ctr < AR].mean(1); fire_peak = tun_r.max(1) + 1e-9
com = (tun_r * ctr[None, :]).sum(1) / np.maximum(tun_r.sum(1), 1e-9)
def ivt50(c):
    r = tun_r[c]; idx = np.where(r >= 0.5 * r.max())[0]; return ctr[idx[0]] if len(idx) else np.nan
def klass(c):
    if regfire[c] < 150: return "rare"
    if fire_peak[c] < 1e-3: return "silent"
    if fire_dry[c] > 0.4 * fire_peak[c]: return "non-AR"
    return "AR-intensity" if ivt50(c) >= 600 else "AR-presence"
classes = np.array([klass(c) for c in range(4096)])
reg = np.where(regfrac > 0.15)[0]
for lab in ["AR-presence", "AR-intensity", "non-AR"]:
    sub = [c for c in reg if classes[c] == lab]; sub.sort(key=lambda c: -rate[c])
    print(f"\n=== W_S_America {lab}  ({len(sub)} regional) ===")
    print(f"{'con':>5}{'rate':>9}{'regfrac':>8}{'com_ivt':>8}{'ivt50':>7}{'regfire':>8}")
    for c in sub[:15]:
        print(f"{c:>5}{rate[c]:>9.4f}{regfrac[c]:>8.2f}{com[c]:>8.0f}{ivt50(c):>7.0f}{int(regfire[c]):>8}")
