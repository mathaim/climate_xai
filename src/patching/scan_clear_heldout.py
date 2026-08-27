"""EXHAUSTIVE scan of every held-out summer timestep (2018-2021, all 6h) from the WeatherBench2 zarr."""
import numpy as np, gcsfs, xarray as xr
from src.analysis.ar_intensity.ivt import ivt
fs = gcsfs.GCSFileSystem(token="anon")
ds = xr.open_zarr(fs.get_mapper("weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"), consolidated=True)
if "latitude" in ds.coords: ds = ds.rename(latitude="lat")
if "longitude" in ds.coords: ds = ds.rename(longitude="lon")
if float(ds.lat[0]) > float(ds.lat[-1]): ds = ds.reindex(lat=ds.lat[::-1])
w = np.load("/scratch/euh7ys/climate_xai/patching/inject_field_1592.npz")
LO0, LO1 = float(w["lon"].min()) % 360, float(w["lon"].max()) % 360
sub = ds.sel(lat=slice(25.0, 58.0), lon=slice(LO0, LO1))
lev = np.asarray(ds["level"].values, float); L = len(lev)
tstr = set(ds.time.values.astype("datetime64[h]").astype(str))
cands = [np.datetime64(f"{y}-{mo:02d}-{day:02d}T{hh:02d}:00:00")
         for y in (2018, 2019, 2020, 2021) for mo in (6, 7, 8, 9) for day in range(1, 31) for hh in (0, 6, 12, 18)]
print(f"scanning up to {len(cands)} held-out summer 6h timesteps", flush=True)
rows = []
for k, vt in enumerate(cands):
    if str(vt.astype("datetime64[h]")) not in tstr: continue
    s = sub.sel(time=vt)
    q = s["specific_humidity"].values; u = s["u_component_of_wind"].values; v = s["v_component_of_wind"].values
    nlat, nlon = q.shape[1], q.shape[2]
    iv = ivt(q.reshape(L, -1).T, u.reshape(L, -1).T, v.reshape(L, -1).T, lev).reshape(nlat, nlon)
    rows.append((float(np.nanmax(iv)), float(np.nanpercentile(iv, 99)), str(vt)))
    if (k + 1) % 100 == 0: print(f"  {k+1}/{len(cands)}", flush=True)
nunder = sum(1 for r in rows if r[0] < 250)
print(f"\n{nunder}/{len(rows)} held-out summer timesteps have window-max IVT < 250", flush=True)
print("=== quietest held-out windows (2018-2021, exhaustive) ===", flush=True)
for mx, p99, s in sorted(rows)[:25]:
    tgt = str(np.datetime64(s) - np.timedelta64(6, "h"))
    print(f"valid {s}  max {mx:5.0f}  p99 {p99:4.0f}   TARGET={tgt}", flush=True)
print("DONE", flush=True)
