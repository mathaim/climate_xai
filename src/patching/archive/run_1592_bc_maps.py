"""Five-panel data for 1592 on the BC AR: ERA5 truth + baseline + beta edits.
Saves NE-Pacific IVT fields to bc_maps_1592.npz. Cross-check scalars against bc_ar_1592.csv.
Env: TARGET (valid time, default 2021-11-15T12:00), BETAS (default "0,0.5,1.5")."""
import os, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.analysis.ar_intensity.ivt import ivt
D = "/scratch/euh7ys/climate_xai/patching"
TARGET = os.environ.get("TARGET", "2021-11-15T12:00")
BETAS = [float(x) for x in os.environ.get("BETAS", "0,0.5,1.5").split(",")]

def pred_ivt_window(pred, la, lo):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    lv = np.asarray(q["level"].values, float); L = len(lv)
    lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    iv = ivt(q.values[0,0].reshape(L,-1).T, u.values[0,0].reshape(L,-1).T,
             v.values[0,0].reshape(L,-1).T, lv).reshape(len(lat), len(lon))
    lam = np.isin(np.round(lat,3), np.round(la,3)); lom = np.isin(np.round(lon,3), np.round(lo,3))
    return iv[np.ix_(lam, lom)].astype(np.float32)

def main():
    win = np.load(f"{D}/inject_field_1592.npz")            # reuse the NE-Pacific window
    la, lo = win["lat"], win["lon"]
    out = {"lat": la, "lon": lo}

    # --- ERA5 ground truth IVT at the valid time, same window ---
    ds = P._get_zarr()
    t = np.datetime64(TARGET.replace("T", " ").replace(" ", "T")) + np.timedelta64(6, "h")  # forecast verifies at T+6h
    sel = ds.sel(time=t)
    lv = np.asarray(sel["level"].values, float); L = len(lv)
    lat = sel["latitude"].values if "latitude" in sel.coords else sel["lat"].values
    lon = sel["longitude"].values if "longitude" in sel.coords else sel["lon"].values
    lon = ((lon + 180) % 360) - 180
    q = sel["specific_humidity"].values; u = sel["u_component_of_wind"].values; v = sel["v_component_of_wind"].values
    iv = ivt(q.reshape(L,-1).T, u.reshape(L,-1).T, v.reshape(L,-1).T, lv).reshape(len(lat), len(lon))
    lam = np.isin(np.round(lat,3), np.round(la,3)); lom = np.isin(np.round(lon,3), np.round(lo,3))
    out["truth"] = iv[np.ix_(lam, lom)].astype(np.float32)
    print(f"truth: window {out['truth'].shape} peak {out['truth'].max():.0f}", flush=True)

    # --- forecasts: baseline + betas (alpha offset = beta - 1) ---
    S = P.setup(f"{D}/plain_L8_sae.npz")
    inp, tar, frc = P.build_inputs(TARGET, S)
    base = P.run_one(P.make_forward(None, S, step=8), inp, tar, frc)
    out["baseline"] = pred_ivt_window(base, la, lo)
    print(f"baseline: peak {out['baseline'].max():.0f} (csv a+0.0 at {TARGET}: compare!)", flush=True)
    for b in BETAS:
        a = np.zeros(4096, np.float32); a[1592] = b - 1.0
        pred = P.run_one(P.make_forward(jnp.asarray(a), S, step=8), inp, tar, frc)
        out[f"beta{b:g}"] = pred_ivt_window(pred, la, lo)
        print(f"beta={b:g}: peak {out[f'beta{b:g}'].max():.0f}", flush=True)

    np.savez(f"{D}/{os.environ.get('OUTNPZ', 'bc_maps_1592.npz')}", **out)
    print("saved", os.environ.get("OUTNPZ", "bc_maps_1592.npz"), " ALL DONE", flush=True)

if __name__ == "__main__":
    main()
