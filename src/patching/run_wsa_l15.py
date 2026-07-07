"""Lens 3: inject each 340-family delta at L8, capture the L15 mesh activation + output box IVT.
Baseline (zeros) + clamp per concept. Saves L15 activations for offline matry_L15 SAE encode."""
import os, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.analysis.ar_intensity.ivt import ivt
PATCH = "/scratch/euh7ys/climate_xai/patching"; NPZ = f"{PATCH}/plain_L8_sae.npz"
LAT = (-50, -30); LON = (-77, -62)
def box_ivt(pred):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    levels = np.asarray(q["level"].values, float); lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    la = (lat >= LAT[0]) & (lat <= LAT[1]); lo = (lon >= LON[0]) & (lon <= LON[1]); L = len(levels)
    qs = q.values[0, 0][:, la][:, :, lo]; us = u.values[0, 0][:, la][:, :, lo]; vs = v.values[0, 0][:, la][:, :, lo]
    iv = ivt(qs.reshape(L, -1).T, us.reshape(L, -1).T, vs.reshape(L, -1).T, levels)
    return float(np.nanmax(iv)), float(np.nanmean(iv))
def run(tag, field, S, inp, tar, frc):
    holder = []
    fwd = P.make_field_capture_forward(jnp.asarray(field), S, holder, inject_step=8, capture_step=15)
    pred = P.run_one(fwd, inp, tar, frc); mx, mn = box_ivt(pred)
    l15 = np.asarray(holder[0], np.float32) if holder else None
    if l15 is not None: np.save(f"{PATCH}/l15_cap_{tag}.npy", l15)
    print(f"{tag}: box_max {mx:.1f}  box_mean {mn:.1f}  l15 {None if l15 is None else l15.shape}", flush=True)
def main():
    S = P.setup(NPZ); meta = np.load(f"{PATCH}/patch_meta.npz", allow_pickle=True); T = str(meta["target_time"])
    print("target_time", T, flush=True); inp, tar, frc = P.build_inputs(T, S); z = None
    for cc in [3481, 3948, 3675, 340]:
        fp = f"{PATCH}/delta_clamp_{cc}.npy"
        if not os.path.exists(fp): print("skip (no delta)", cc, flush=True); continue
        d = np.load(fp).astype(np.float32)
        if z is None: z = np.zeros_like(d)[:, None, :]; run("baseline", z, S, inp, tar, frc)
        run(f"clamp_{cc}", d[:, None, :], S, inp, tar, frc)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
