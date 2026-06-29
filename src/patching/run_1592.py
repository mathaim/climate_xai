"""Dose-response: clamp -> baseline -> amplify Plain-L8 concept 1592 over the BC AR strong phase."""
import os, numpy as np, jax.numpy as jnp, pandas as pd
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.patching.region_ivt_pred import region_ivt
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt/track_pool_W_N_America.npz"
OUT = "/scratch/euh7ys/climate_xai/patching"
C, LAYER = 1592, 8
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]          # clamp ... baseline ... amplify
TIMES = ["2021-11-13T18:00", "2021-11-14T00:00", "2021-11-14T06:00", "2021-11-15T06:00", "2021-11-15T12:00"]
def avec(c, a):
    v = np.zeros(latent_dim(), np.float32); v[c] = a; return jnp.asarray(v)
def main():
    os.makedirs(OUT, exist_ok=True); S = P.setup(NPZ)
    d = np.load(TRACK); A = d["A_max"].astype(np.float32); iv = d["ivt"].astype(float); ok = np.isfinite(iv)
    A = A[ok]; iv = iv[ok]; fires = (A > 0).mean(0)
    a_ = A - A.mean(0); yi = iv - iv.mean()
    r = np.abs((a_ * yi[:, None]).sum(0) / np.sqrt((a_ ** 2).sum(0) * (yi ** 2).sum() + 1e-12))
    cand = np.where(fires > 0.05)[0]; ctrl = int(cand[np.argmin(r[cand])])
    print(f"concept {C}: |r|(IVT)={r[C]:.2f}  fires={fires[C]:.0%}  |  control {ctrl}: |r|={r[ctrl]:.2f}", flush=True)
    fwd = {f"c{C}_a{a:+.1f}": P.make_forward(avec(C, a), S, LAYER) for a in ALPHAS}
    fwd.update({f"ctrl_a{a:+.1f}": P.make_forward(avec(ctrl, a), S, LAYER) for a in ALPHAS})
    rows = []
    for T in TIMES:
        inp, tar, frc = P.build_inputs(T, S)
        for name, fn in fwd.items():
            mx, mn = region_ivt(P.run_one(fn, inp, tar, frc))
            rows.append({"time": T, "cond": name, "ivt_max": mx}); print(T, name, round(mx, 1), flush=True)
        del inp, tar, frc
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/bc_ar_1592.csv", index=False)
    piv = df.pivot(index="time", columns="cond", values="ivt_max"); base = piv[f"c{C}_a+0.0"]
    print("\n--- mean region-max-IVT change vs baseline (alpha=0) ---")
    for c in piv.columns:
        print(f"  {c:14s}  {float((piv[c] - base).mean()):+8.1f}")
    print("DONE")
if __name__ == "__main__":
    main()
