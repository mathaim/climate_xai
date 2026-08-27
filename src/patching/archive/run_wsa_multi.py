"""Gain sweep of child 3481 across dry/moderate/strong S-Chile timesteps: landfall IVT vs gain g."""
import numpy as np, jax.numpy as jnp, pandas as pd
from src.patching import patch_predict as P
from src.analysis.ar_intensity.ivt import ivt
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
PATCH = "/scratch/euh7ys/climate_xai/patching"; LAYER = 8
LAT = (-50, -30); LON = (-77, -62); GAINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
def ivt_metrics(pred, core_ll):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    levels = np.asarray(q["level"].values, float); lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    la = (lat >= LAT[0]) & (lat <= LAT[1]); lo = (lon >= LON[0]) & (lon <= LON[1]); L = len(levels)
    qs = q.values[0, 0][:, la][:, :, lo]; us = u.values[0, 0][:, la][:, :, lo]; vs = v.values[0, 0][:, la][:, :, lo]
    iv = ivt(qs.reshape(L, -1).T, us.reshape(L, -1).T, vs.reshape(L, -1).T, levels)
    ilat = int(np.argmin(np.abs(lat - core_ll[0]))); ilon = int(np.argmin(np.abs(lon - core_ll[1])))
    qc = q.values[0, 0][:, ilat, ilon]; uc = u.values[0, 0][:, ilat, ilon]; vc = v.values[0, 0][:, ilat, ilon]
    return float(np.nanmax(iv)), float(np.nanmean(iv)), float(ivt(qc[None, :], uc[None, :], vc[None, :], levels)[0])
def main():
    S = P.setup(NPZ); meta = np.load(f"{PATCH}/multi_meta.npz", allow_pickle=True)
    core_ll = (float(meta["core_lat"]), ((float(meta["core_lon"]) + 180) % 360) - 180); rows = []
    for tag in ["off", "low", "high"]:
        T = str(meta[f"time_{tag}"]); ivb = float(meta[f"iv_{tag}"])
        base = np.load(f"{PATCH}/base_{tag}.npy").astype(np.float32)
        inp, tar, frc = P.build_inputs(T, S)
        print(f"=== {tag}  {T}  input coreIVT {ivb:.0f} ===", flush=True)
        for g in GAINS:
            mx, mn, cv = ivt_metrics(P.run_one(P.make_field_forward(jnp.asarray(((g - 1.0) * base)[:, None, :]), S, LAYER), inp, tar, frc), core_ll)
            print(f"  {tag} g {g:.1f}  core {cv:.1f}  box_max {mx:.1f}", flush=True)
            rows.append({"tag": tag, "time": T, "input_ivt": ivb, "gain": g, "core": cv, "box_max": mx, "box_mean": mn})
        del inp, tar, frc
    pd.DataFrame(rows).to_csv(f"{PATCH}/wsa_multi.csv", index=False); print("DONE", flush=True)
if __name__ == "__main__":
    main()
