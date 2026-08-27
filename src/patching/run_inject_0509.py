"""July injections at code values 0.3 and 0.9 (= 0.5x and 1.5x characteristic 0.6).
Adds windowed IVT fields to july_maps_1592.npz alongside truth/baseline."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.patching.region_ivt_pred import region_ivt
from src.patching.run_1592_bc_maps import pred_ivt_window
D = "/scratch/euh7ys/climate_xai/patching"; C, LAYER, T = 1592, 8, "2021-07-15T00:00"
def main():
    d = dict(np.load(f"{D}/july_maps_1592.npz"))
    S = P.setup(f"{D}/plain_L8_sae.npz")
    inp, tar, frc = P.build_inputs(T, S)
    for b in (0.3, 0.9):
        v = np.zeros(latent_dim(), np.float32); v[C] = b
        pred = P.run_one(P.make_forward(jnp.asarray(v), S, LAYER, injector_cls=P.AddInjector), inp, tar, frc)
        mx, mn = region_ivt(pred)
        d[f"inject_b{b:g}"] = pred_ivt_window(pred, d["lat"], d["lon"])
        print(f"b={b}: region max {mx:.0f} mean {mn:.0f}", flush=True)
    old = np.load(f"{D}/inject_field_1592.npz")
    d["inject_b0.6"] = old["inject_b0.6"]          # beta=1.0 panel, from the original run
    np.savez(f"{D}/july_maps_1592.npz", **d)
    print("saved july_maps_1592.npz  ALL DONE", flush=True)
if __name__ == "__main__":
    main()
