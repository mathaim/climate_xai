"""Clear-day SUFFICIENCY maps: additively inject Plain-L8 concept 1592 (global AddInjector) into a
genuinely clear day; save ABSOLUTE IVT window fields. Env: TARGET, OUTNPZ, BVALS (default 0.6,1.0,1.5)."""
import os, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.analysis.ar_intensity.ivt import ivt
D = "/scratch/euh7ys/climate_xai/patching"
TARGET = os.environ.get("TARGET", "1994-06-20T06:00")
BVALS = [float(x) for x in os.environ.get("BVALS", "0.6,1.0,1.5").split(",")]
C, LAYER = 1592, 8
def bvec(b):
    v = np.zeros(latent_dim(), np.float32); v[C] = b; return jnp.asarray(v)
def pred_ivt_window(pred, la, lo):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    lv = np.asarray(q["level"].values, float); L = len(lv)
    lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    iv = ivt(q.values[0,0].reshape(L,-1).T, u.values[0,0].reshape(L,-1).T,
             v.values[0,0].reshape(L,-1).T, lv).reshape(len(lat), len(lon))
    lam = np.isin(np.round(lat,3), np.round(la,3)); lom = np.isin(np.round(lon,3), np.round(lo,3))
    return iv[np.ix_(lam, lom)].astype(np.float32)
def main():
    win = np.load(f"{D}/inject_field_1592.npz"); la, lo = win["lat"], win["lon"]
    out = {"lat": la, "lon": lo, "bvals": np.asarray(BVALS, np.float32)}
    ds = P._get_zarr(); t = np.datetime64(TARGET) + np.timedelta64(6, "h"); sel = ds.sel(time=t)
    lv = np.asarray(sel["level"].values, float); L = len(lv)
    lat = sel["lat"].values; lon = ((sel["lon"].values + 180) % 360) - 180
    q = sel["specific_humidity"].values; u = sel["u_component_of_wind"].values; v = sel["v_component_of_wind"].values
    iv = ivt(q.reshape(L,-1).T, u.reshape(L,-1).T, v.reshape(L,-1).T, lv).reshape(len(lat), len(lon))
    lam = np.isin(np.round(lat,3), np.round(la,3)); lom = np.isin(np.round(lon,3), np.round(lo,3))
    out["truth"] = iv[np.ix_(lam, lom)].astype(np.float32); print(f"truth peak {out['truth'].max():.0f}", flush=True)
    S = P.setup(f"{D}/plain_L8_sae.npz"); inp, tar, frc = P.build_inputs(TARGET, S)
    out["baseline"] = pred_ivt_window(P.run_one(P.make_forward(None, S, step=LAYER), inp, tar, frc), la, lo)
    print(f"baseline peak {out['baseline'].max():.0f}", flush=True)
    for i, b in enumerate(BVALS, 1):
        pred = P.run_one(P.make_forward(bvec(b), S, LAYER, injector_cls=P.AddInjector), inp, tar, frc)
        out[f"inj{i}"] = pred_ivt_window(pred, la, lo); print(f"inj{i} (b={b}) peak {out[f'inj{i}'].max():.0f}", flush=True)
    np.savez(f"{D}/{os.environ.get('OUTNPZ','clear_maps_1592.npz')}", **out)
    print("saved", os.environ.get("OUTNPZ", "clear_maps_1592.npz"), "ALL DONE", flush=True)
if __name__ == "__main__":
    main()
