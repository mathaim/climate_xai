"""Clamp 1592 to zero on the clear July 2021 day; save the NE-Pacific IVT field into the
existing inject_field_1592.npz alongside baseline and injection fields."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.analysis.ar_intensity.ivt import ivt
D = "/scratch/euh7ys/climate_xai/patching"
def main():
    d = dict(np.load(f"{D}/inject_field_1592.npz"))
    S = P.setup(f"{D}/plain_L8_sae.npz")
    inp, tar, frc = P.build_inputs("2021-07-15T00:00", S)
    a = np.zeros(4096, np.float32); a[1592] = -1.0
    pred = P.run_one(P.make_forward(jnp.asarray(a), S, step=8), inp, tar, frc)
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    lv = np.asarray(q["level"].values, float); L = len(lv)
    lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    iv = ivt(q.values[0,0].reshape(L,-1).T, u.values[0,0].reshape(L,-1).T, v.values[0,0].reshape(L,-1).T, lv).reshape(len(lat), len(lon))
    la = np.isin(np.round(lat, 3), np.round(d["lat"], 3)); lo = np.isin(np.round(lon, 3), np.round(d["lon"], 3))
    d["clamp_a1.0"] = iv[np.ix_(la, lo)].astype(np.float32)
    np.savez(f"{D}/inject_field_1592.npz", **d)
    print("clamp field saved; window", d["clamp_a1.0"].shape, "vs baseline", d["clear"].shape, flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
