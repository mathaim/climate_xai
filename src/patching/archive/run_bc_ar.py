"""Clamp Plain-L15 concept 2251 across the Nov 2021 BC AR; predicted region IVT vs baseline + control."""
import os, numpy as np, jax.numpy as jnp, pandas as pd
from datetime import datetime, timedelta
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.patching.region_ivt_pred import region_ivt
OUT = "/scratch/euh7ys/climate_xai/patching"; CLAMP = 2251
def times():
    t = datetime(2021, 11, 13, 0)
    return [(t + timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M") for i in range(16)]
def alpha_vec(idx):
    a = np.zeros(latent_dim(), np.float32)
    if idx is not None: a[idx] = -1.0
    return jnp.asarray(a)
def pick_control():
    d = np.load("/scratch/euh7ys/climate_xai/concept_ivt/track_plain_L15_W_N_America.npz")
    A = d["A_max"].astype(np.float32); ivt = d["ivt"].astype(float); ok = np.isfinite(ivt)
    A = A[ok]; iv = ivt[ok]; fires = (A > 0).mean(0)
    a = A - A.mean(0); yi = iv - iv.mean()
    r = np.abs((a * yi[:, None]).sum(0) / np.sqrt((a ** 2).sum(0) * (yi ** 2).sum() + 1e-12))
    cand = np.where(fires > 0.05)[0]
    c = int(cand[np.argmin(r[cand])]); print(f"control concept {c} (|r|={r[c]:.3f}, fires {fires[c]:.0%})", flush=True)
    return c
def main():
    os.makedirs(OUT, exist_ok=True); S = P.setup(); CONTROL = pick_control()
    f = {"baseline": P.make_forward(alpha_vec(None), S),   # alpha=0 (injector identity)
         "clamp":    P.make_forward(alpha_vec(CLAMP), S),
         "control":  P.make_forward(alpha_vec(CONTROL), S)}
    rows = []
    for T in times():
        for name, fn in f.items():
            mx, mn = region_ivt(P.predict(T, fn, S))
            rows.append({"time": T, "cond": name, "ivt_max": mx, "ivt_mean": mn})
            print(T, name, round(mx, 1), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/bc_ar_2251.csv", index=False)
    piv = df.pivot(index="time", columns="cond", values="ivt_max")
    piv["dIVT_clamp"] = piv["clamp"] - piv["baseline"]; piv["dIVT_control"] = piv["control"] - piv["baseline"]
    print(piv.round(1).to_string()); print("DONE")
if __name__ == "__main__":
    main()
