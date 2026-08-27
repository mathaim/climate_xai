"""Capstone injections: 99 clamp/amplify (global AR-intensity dial) and 340 amplify
(dry out Chile), all with L15 capture + box and global IVT readout."""
import os, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.analysis.ar_intensity.ivt import ivt
PATCH = "/scratch/euh7ys/climate_xai/patching"; NPZ = f"{PATCH}/plain_L8_sae.npz"
LAT = (-50, -30); LON = (-77, -62)
def metrics(pred):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    lv = np.asarray(q["level"].values, float); L = len(lv)
    lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    iv = ivt(q.values[0,0].reshape(L,-1).T, u.values[0,0].reshape(L,-1).T, v.values[0,0].reshape(L,-1).T, lv).reshape(len(lat), len(lon))
    la = (lat >= LAT[0]) & (lat <= LAT[1]); lo = (lon >= LON[0]) & (lon <= LON[1])
    return float(np.nanmean(iv)), float(np.nanmax(iv)), float(np.nanmean(iv[np.ix_(la, lo)])), float(np.nanmax(iv[np.ix_(la, lo)])), iv
def run(tag, field, S, inp, tar, frc, save_map=False):
    holder = []
    fwd = P.make_field_capture_forward(jnp.asarray(field), S, holder, inject_step=8, capture_step=15)
    pred = P.run_one(fwd, inp, tar, frc)
    gm, gx, bm, bx, iv = metrics(pred)
    if holder: np.save(f"{PATCH}/l15_cap_{tag}.npy", np.asarray(holder[0], np.float32))
    if save_map: np.save(f"{PATCH}/ivtmap_{tag}.npy", iv.astype(np.float32))
    print(f"{tag}: global IVT mean {gm:.1f} max {gx:.1f} | Chile box mean {bm:.1f} max {bx:.1f}", flush=True)
def main():
    S = P.setup(NPZ); meta = np.load(f"{PATCH}/patch_meta.npz", allow_pickle=True); T = str(meta["target_time"])
    print("target_time", T, flush=True); inp, tar, frc = P.build_inputs(T, S)
    d99 = np.load(f"{PATCH}/delta_clamp_99.npy").astype(np.float32)[:, None, :]
    d340 = np.load(f"{PATCH}/delta_clamp_340.npy").astype(np.float32)[:, None, :]
    run("cap_base", np.zeros_like(d99), S, inp, tar, frc, save_map=True)
    run("99_clamp", d99, S, inp, tar, frc)
    for g in (2, 3): run(f"99_amp{g}", -(g-1)*d99, S, inp, tar, frc, save_map=(g==3))
    for g in (2, 3): run(f"340_amp{g}", -(g-1)*d340, S, inp, tar, frc, save_map=(g==3))
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
