"""Save the wide IVT field for clear air vs 1592-injected, to map the conjured atmospheric river."""
import os, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
from src.analysis.ar_intensity.ivt import ivt
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
OUT = "/scratch/euh7ys/climate_xai/patching"
T, C, LAYER = "2021-07-15T00:00", 1592, 8
LAT0, LAT1, LON0, LON1 = 20, 60, -165, -105     # NE Pacific into W. North America
def field(pred):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    levels = np.asarray(q["level"].values, float)
    lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    la = (lat >= LAT0) & (lat <= LAT1); lo = (lon >= LON0) & (lon <= LON1)
    nlat, nlon = int(la.sum()), int(lo.sum())
    qs = q.values[0, 0][:, la][:, :, lo]; us = u.values[0, 0][:, la][:, :, lo]; vs = v.values[0, 0][:, la][:, :, lo]
    iv = ivt(qs.reshape(len(levels), -1).T, us.reshape(len(levels), -1).T, vs.reshape(len(levels), -1).T, levels)
    return iv.reshape(nlat, nlon), lat[la], lon[lo]
def bvec(b):
    v = np.zeros(latent_dim(), np.float32); v[C] = b; return jnp.asarray(v)
def main():
    os.makedirs(OUT, exist_ok=True); S = P.setup(NPZ)
    inp, tar, frc = P.build_inputs(T, S); saved = {}
    for name, b in [("clear", 0.0), ("inject_b0.6", 0.6), ("inject_b1.0", 1.0)]:
        fn = P.make_forward(bvec(b), S, LAYER, injector_cls=P.AddInjector)
        iv, lat, lon = field(P.run_one(fn, inp, tar, frc)); saved[name] = iv
        print(name, "max region-window IVT", round(float(np.nanmax(iv)), 1), flush=True)
    np.savez(f"{OUT}/inject_field_1592.npz", lat=lat, lon=lon, **saved)
    print("saved", f"{OUT}/inject_field_1592.npz"); print("DONE")
if __name__ == "__main__":
    main()
