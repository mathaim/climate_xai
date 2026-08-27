"""Localize AR-mask temporal alignment. Uses the TRUE (unshifted) labels. If early years show
IVT@AR >> IVT@noAR but later years reverse, the concatenated mask index has drifted from the date."""
import glob, datetime as DT
from collections import defaultdict
import numpy as np
from src.analysis.ar_intensity.ivt_pipeline import region_node_setup, load_channel_index, node_ivt, NPZ, ERA5_DIR
from src.analysis.ar_intensity.regions import REGIONS, AR_START

idx,levels,qi,ui,vi,lat_i,lon_i=load_channel_index()
setup=region_node_setup(); d=np.load(NPZ); masks={r:d[f"{r}__mask"] for r in REGIONS}
print("mask shapes:", {r:tuple(masks[r].shape) for r in REGIONS}, " (expected T ~ 56700)")
files=sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))
first=files[0].split("era5_inputs_")[-1][:10]; last=files[-1].split("era5_inputs_")[-1][:10]
print(f"activation files: {len(files)}   dates {first} .. {last}")
rng=np.random.default_rng(0); sel=[files[i] for i in rng.choice(len(files),800,replace=False)]
agg={r:defaultdict(lambda:{"ar":[],"no":[],"f":[]}) for r in REGIONS}
for f in sel:
    b=f.split("era5_inputs_")[-1].replace(".npy","")
    dt=DT.datetime.strptime(b,"%Y-%m-%dT%H-%M"); ti=int(round((dt-AR_START).total_seconds()/21600))+1
    arr=np.load(f,mmap_mode="r")
    for r in REGIONS:
        if not (0<=ti-1<masks[r].shape[0]): continue
        nodes=setup[r]["nodes"]; iv=np.asarray(node_ivt(np.ascontiguousarray(arr[nodes]),qi,ui,vi,levels))
        lab=masks[r][ti-1][setup[r]["li"],setup[r]["ji"]]>0; yb=(dt.year//5)*5
        agg[r][yb]["f"].append(float(lab.mean()))
        if lab.any(): agg[r][yb]["ar"].append(float(iv[lab].mean()))
        if (~lab).any(): agg[r][yb]["no"].append(float(iv[~lab].mean()))
for r in REGIONS:
    print(f"\n{r}:  (IVT@AR vs IVT@noAR by 5-yr window; AR should be higher)")
    for yb in sorted(agg[r]):
        A=agg[r][yb]; a=np.mean(A["ar"]) if A["ar"] else float("nan"); n=np.mean(A["no"]) if A["no"] else float("nan")
        print(f"  {yb}-{yb+4}: AR={a:5.0f}  noAR={n:5.0f}  ARfrac={np.mean(A['f']):.2f}  n={len(A['f']):3d}   {'OK' if a>n else 'BACKWARDS'}")
