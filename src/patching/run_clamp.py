"""Clamp top-|r| Plain-L{LAYER} AR concepts (single + set) on the BC AR vs control; region IVT.
Layer via CLAMP_LAYER env (default 8)."""
import os, numpy as np, jax.numpy as jnp, pandas as pd
from datetime import datetime, timedelta
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.patching.region_ivt_pred import region_ivt
LAYER = int(os.environ.get("CLAMP_LAYER", "8")); K = 10
NPZ = f"/scratch/euh7ys/climate_xai/patching/plain_L{LAYER}_sae.npz"
TRACK = f"/scratch/euh7ys/climate_xai/concept_ivt/track_plain_L{LAYER}_W_N_America.npz"
OUT = "/scratch/euh7ys/climate_xai/patching"
def times():
    t = datetime(2021, 11, 13, 0)
    return [(t + timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M") for i in range(16)]
def corr_fires():
    d = np.load(TRACK); A = d["A_max"].astype(np.float32); ivt = d["ivt"].astype(float); ok = np.isfinite(ivt)
    A = A[ok]; iv = ivt[ok]; fires = (A > 0).mean(0)
    a = A - A.mean(0); yi = iv - iv.mean()
    r = np.abs((a * yi[:, None]).sum(0) / np.sqrt((a ** 2).sum(0) * (yi ** 2).sum() + 1e-12))
    return r, fires
def alpha_set(idxs):
    a = np.zeros(latent_dim(), np.float32)
    if len(idxs): a[list(idxs)] = -1.0
    return jnp.asarray(a)
def main():
    os.makedirs(OUT, exist_ok=True)
    r, fires = corr_fires(); order = np.argsort(r)[::-1]
    cand = np.where(fires > 0.05)[0]; co = cand[np.argsort(r[cand])]
    conds = {"baseline": [], "clamp_single": [int(order[0])], "control_single": [int(co[0])],
             "clamp_set": order[:K].tolist(), "control_set": co[:K].tolist()}
    print(f"L{LAYER} clamp_single {conds['clamp_single']} r={r[conds['clamp_single']].round(2).tolist()}", flush=True)
    print(f"L{LAYER} clamp_set    {conds['clamp_set']} r={r[conds['clamp_set']].round(2).tolist()}", flush=True)
    print(f"L{LAYER} control_single {conds['control_single']}  control_set {conds['control_set']}", flush=True)
    S = P.setup(NPZ)
    fwd = {name: P.make_forward(alpha_set(idx), S, LAYER) for name, idx in conds.items()}
    rows = []
    for T in times():
        for name, fn in fwd.items():
            mx, mn = region_ivt(P.predict(T, fn, S))
            rows.append({"time": T, "cond": name, "ivt_max": mx, "ivt_mean": mn})
            print(T, name, round(mx, 1), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/bc_ar_L{LAYER}.csv", index=False)
    piv = df.pivot(index="time", columns="cond", values="ivt_max")
    for c in ["clamp_single", "control_single", "clamp_set", "control_set"]:
        piv["d_" + c] = piv[c] - piv["baseline"]
    print(piv[["baseline", "d_clamp_single", "d_control_single", "d_clamp_set", "d_control_set"]].round(1).to_string())
    print("DONE")
if __name__ == "__main__":
    main()
