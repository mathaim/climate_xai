"""Matryoshka causal patch: clamp parent 340 vs child 3481 at step 8, read predicted W_S_America IVT."""
import numpy as np, jax.numpy as jnp, pandas as pd
from src.patching import patch_predict as P
from src.analysis.ar_intensity.ivt import ivt
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"   # only for setup(); FieldInjector ignores it
PATCH = "/scratch/euh7ys/climate_xai/patching"; LAYER = 8
LAT = (-50, -30); LON = (-77, -62)
def ivt_metrics(pred, core_ll):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    levels = np.asarray(q["level"].values, float)
    lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    la = (lat >= LAT[0]) & (lat <= LAT[1]); lo = (lon >= LON[0]) & (lon <= LON[1]); L = len(levels)
    qs = q.values[0, 0][:, la][:, :, lo]; us = u.values[0, 0][:, la][:, :, lo]; vs = v.values[0, 0][:, la][:, :, lo]
    iv = ivt(qs.reshape(L, -1).T, us.reshape(L, -1).T, vs.reshape(L, -1).T, levels)
    ilat = int(np.argmin(np.abs(lat - core_ll[0]))); ilon = int(np.argmin(np.abs(lon - core_ll[1])))
    qc = q.values[0, 0][:, ilat, ilon]; uc = u.values[0, 0][:, ilat, ilon]; vc = v.values[0, 0][:, ilat, ilon]
    ivc = float(ivt(qc[None, :], uc[None, :], vc[None, :], levels)[0])
    return float(np.nanmax(iv)), float(np.nanmean(iv)), ivc
def field(cc):
    d = np.load(f"{PATCH}/delta_clamp_{cc}.npy").astype(np.float32); return jnp.asarray(d[:, None, :])
def main():
    S = P.setup(NPZ)
    meta = np.load(f"{PATCH}/patch_meta.npz", allow_pickle=True)
    T = str(meta["target_time"]); x8 = meta["x8"].astype(np.float32)
    core_ll = (float(meta["core_lat"]), ((float(meta["core_lon"]) + 180) % 360) - 180)
    print("target_time", T, "core", core_ll, flush=True)
    inp, tar, frc = P.build_inputs(T, S)
    print("--- gate (expect small rel) ---", flush=True)
    P.run_one(P.make_field_forward(None, S, LAYER, gate_x8=jnp.asarray(x8[:, None, :])), inp, tar, frc)
    mx0, mn0, c0 = ivt_metrics(P.run_one(P.make_forward(None, S, LAYER), inp, tar, frc), core_ll)
    print(f"baseline    box_max {mx0:.1f}  box_mean {mn0:.1f}  core {c0:.1f}", flush=True)
    rows = [{"cond": "baseline", "box_max": mx0, "box_mean": mn0, "core": c0}]
    for cc in [340, 3481]:
        mx, mn, cv = ivt_metrics(P.run_one(P.make_field_forward(field(cc), S, LAYER), inp, tar, frc), core_ll)
        print(f"clamp {cc:>4}   box_max {mx:.1f} ({mx-mx0:+.1f})  box_mean {mn:.1f} ({mn-mn0:+.1f})  core {cv:.1f} ({cv-c0:+.1f})", flush=True)
        rows.append({"cond": f"clamp_{cc}", "box_max": mx, "box_mean": mn, "core": cv})
    pd.DataFrame(rows).to_csv(f"{PATCH}/wsa_patch.csv", index=False); print("DONE", flush=True)
if __name__ == "__main__":
    main()
