"""Clamp the SET of top-K IVT-correlated Plain-L15 concepts on the BC AR vs a low-|r| control set."""
import os, numpy as np, jax.numpy as jnp, pandas as pd
from datetime import datetime, timedelta
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.patching.region_ivt_pred import region_ivt
OUT = "/scratch/euh7ys/climate_xai/patching"; K = 10
def times():
    t = datetime(2021, 11, 13, 0)
    return [(t + timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M") for i in range(16)]
def pick_sets():
    d = np.load("/scratch/euh7ys/climate_xai/concept_ivt/track_plain_L15_W_N_America.npz")
    A = d["A_max"].astype(np.float32); ivt = d["ivt"].astype(float); ok = np.isfinite(ivt)
    A = A[ok]; iv = ivt[ok]; fires = (A > 0).mean(0)
    a = A - A.mean(0); yi = iv - iv.mean()
    r = np.abs((a * yi[:, None]).sum(0) / np.sqrt((a ** 2).sum(0) * (yi ** 2).sum() + 1e-12))
    top = np.argsort(r)[::-1][:K]
    cand = np.where(fires > 0.05)[0]; ctrl = cand[np.argsort(r[cand])[:K]]
    print("AR set (top |r|):", top.tolist(), "r:", r[top].round(2).tolist(), flush=True)
    print("control set (low |r|):", ctrl.tolist(), "r:", r[ctrl].round(2).tolist(), flush=True)
    return top.tolist(), ctrl.tolist()
def alpha_set(idxs):
    a = np.zeros(latent_dim(), np.float32); a[idxs] = -1.0; return jnp.asarray(a)
def main():
    os.makedirs(OUT, exist_ok=True); S = P.setup(); top, ctrl = pick_sets()
    f = {"baseline": P.make_forward(alpha_set([]), S),
         "clamp_set": P.make_forward(alpha_set(top), S),
         "control_set": P.make_forward(alpha_set(ctrl), S)}
    rows = []
    for T in times():
        for name, fn in f.items():
            mx, mn = region_ivt(P.predict(T, fn, S))
            rows.append({"time": T, "cond": name, "ivt_max": mx, "ivt_mean": mn})
            print(T, name, round(mx, 1), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/bc_ar_set.csv", index=False)
    piv = df.pivot(index="time", columns="cond", values="ivt_max")
    piv["dIVT_set"] = piv["clamp_set"] - piv["baseline"]; piv["dIVT_ctrl"] = piv["control_set"] - piv["baseline"]
    print(piv.round(1).to_string()); print("DONE")
if __name__ == "__main__":
    main()
