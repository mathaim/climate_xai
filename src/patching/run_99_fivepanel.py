"""Concept-99 five-panel data: ERA5 truth (valid T+6h) + clamped beta=0.5 field.
Baseline / removed(beta0) / amplified(beta3) maps already exist from the ho runs."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.patching.run_capstone import metrics
PATCH = "/scratch/euh7ys/climate_xai/patching"; T = "2021-11-15T12:00"
def main():
    # --- beta=0.5 forward: field = 0.5 * stored (stored = the beta=0 edit) ---
    S = P.setup(f"{PATCH}/plain_L8_sae.npz")
    inp, tar, frc = P.build_inputs(T, S)
    delta = np.load(f"{PATCH}/delta_clamp_99_ho.npy")
    pred = P.run_one(P.make_field_forward(jnp.asarray((0.5 * delta)[:, None, :]), S, 8), inp, tar, frc)
    gm, gx, bm, bx, iv = metrics(pred)
    np.save(f"{PATCH}/ivtmap_ho_99_beta05.npy", iv.astype(np.float32))
    print(f"beta=0.5 map saved; global mean {gm:.1f} max {gx:.1f}", flush=True)

    # --- ERA5 truth at the valid time (T+6h), global, on the same grid layout ---
    from src.analysis.ar_intensity.ivt import ivt
    ds = P._get_zarr()
    t = np.datetime64(T.replace("T", " ").replace(" ", "T")) + np.timedelta64(6, "h")
    sel = ds.sel(time=t)
    lv = np.asarray(sel["level"].values, float); L = len(lv)
    latc = sel["latitude"].values if "latitude" in sel.coords else sel["lat"].values
    q = sel["specific_humidity"].values; u = sel["u_component_of_wind"].values; v = sel["v_component_of_wind"].values
    iv_t = ivt(q.reshape(L,-1).T, u.reshape(L,-1).T, v.reshape(L,-1).T, lv).reshape(len(latc), -1)
    if latc[0] > latc[-1]:            # zarr latitude often descends; match the ascending map layout
        iv_t = iv_t[::-1]
    base = np.load(f"{PATCH}/ivtmap_ho_base.npy")
    print(f"truth shape {iv_t.shape} vs base {base.shape}  truth peak {np.nanmax(iv_t):.0f}", flush=True)
    np.save(f"{PATCH}/ivtmap_ho_truth.npy", iv_t.astype(np.float32))
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
