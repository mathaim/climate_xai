"""Sufficiency test: additively inject Plain-L8 concept 1592 into a clear-air timestep -> does an AR appear?"""
import os, numpy as np, jax.numpy as jnp, pandas as pd
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.patching.region_ivt_pred import region_ivt
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt/track_pool_W_N_America.npz"
OUT = "/scratch/euh7ys/climate_xai/patching"
C, LAYER = 1592, 8
CAND = ["2021-07-15T00:00", "2021-08-12T00:00", "2020-08-05T00:00"]   # summer, expect low IVT
BETAS = [0.0, 0.3, 0.6, 1.0, 2.0]                                     # 0=baseline; ~0.6 = real AR firing level
def bvec(c, b):
    v = np.zeros(latent_dim(), np.float32); v[c] = b; return jnp.asarray(v)
def main():
    os.makedirs(OUT, exist_ok=True); S = P.setup(NPZ)
    d = np.load(TRACK); A = d["A_max"].astype(np.float32); iv = d["ivt"].astype(float); ok = np.isfinite(iv)
    A = A[ok]; iv = iv[ok]; fires = (A > 0).mean(0); a_ = A - A.mean(0); yi = iv - iv.mean()
    r = np.abs((a_ * yi[:, None]).sum(0) / np.sqrt((a_ ** 2).sum(0) * (yi ** 2).sum() + 1e-12))
    ctrl = int(np.where(fires > 0.05)[0][np.argmin(r[np.where(fires > 0.05)[0]])])
    f0 = P.make_forward(bvec(C, 0.0), S, LAYER, injector_cls=P.AddInjector)
    base = {}
    for T in CAND:
        inp, tar, frc = P.build_inputs(T, S); base[T] = region_ivt(P.run_one(f0, inp, tar, frc))[0]
        print("candidate", T, "baseline IVT", round(base[T], 1), flush=True); del inp, tar, frc
    T = min(base, key=base.get); print(f"DRY timestep {T} baseline {base[T]:.1f}  |  control concept {ctrl}", flush=True)
    fwd = {f"c{C}_b{b:.1f}": P.make_forward(bvec(C, b), S, LAYER, injector_cls=P.AddInjector) for b in BETAS}
    fwd["ctrl_b2.0"] = P.make_forward(bvec(ctrl, 2.0), S, LAYER, injector_cls=P.AddInjector)
    rows = []; inp, tar, frc = P.build_inputs(T, S)
    for name, fn in fwd.items():
        mx, mn = region_ivt(P.run_one(fn, inp, tar, frc))
        rows.append({"time": T, "cond": name, "ivt_max": mx, "ivt_mean": mn}); print(name, round(mx, 1), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/inject_1592_dry.csv", index=False); print("DONE")
if __name__ == "__main__":
    main()
