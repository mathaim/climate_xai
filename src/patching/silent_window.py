import numpy as np, glob
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
D="/scratch/euh7ys/climate_xai"
d=np.load(f"{D}/concept_ivt/track_pool_W_N_America.npz"); z=d['A_max'][:,1592]; iv=d['ivt']; ti=d['tindex']
files=sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
dates=np.array([files[int(t)].split('era5_inputs_')[-1].replace('.npy','') for t in ti])
idx2,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
w=np.load(f"{D}/patching/inject_field_1592.npz"); LO0,LO1=float(w['lon'].min()),float(w['lon'].max())
c=np.load(f"{D}/patching/clear_maps_7906.npz"); lat=c['lat']; mm=(lat>=25)&(lat<=58)
print("1979-06-09 injected crop-max 25-58N:", [round(float(np.nanmax(c[k][mm])),0) for k in ['inj1','inj2','inj3']], flush=True)
s=(z<1e-6); oo=np.where(s)[0]; oo=oo[np.argsort(iv[oo])][:60]
box=None; rows=[]
for i in oo:
    ds=dates[i]; a=np.load(f"{ERA5_DIR}/era5_inputs_{ds}.npy")
    if box is None:
        nlat=np.asarray(a[:,lat_i]).ravel(); nlon=((np.asarray(a[:,lon_i]).ravel()+180)%360)-180
        box=(nlat>=25)&(nlat<=58)&(nlon>=LO0)&(nlon<=LO1)
    ivv=np.asarray(node_ivt(a,qi,ui,vi,levels)).ravel()[box]
    rows.append((float(ivv.max()), ds, float(iv[i])))
rows.sort()
print("\nsilent-1592 days by WINDOW max IVT (25-58N):", flush=True)
for mx,ds,rivt in rows[:20]:
    dd,tm=ds.split('T'); tgt=str(np.datetime64(f'{dd}T{tm.replace("-",":")}')-np.timedelta64(6,'h'))
    print(f"  {ds}  window {mx:.0f}  regional {rivt:.0f}  TARGET={tgt}", flush=True)
