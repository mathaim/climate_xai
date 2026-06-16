"""Per-node IVT on the GraphCast mesh, restricted to region AR nodes."""
import glob, numpy as np
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity.ivt import ivt
ERA5_DIR = "/scratch/euh7ys/climate_xai/era5_inputs"
NPZ = "/scratch/euh7ys/climate_xai/ar_region_masks.npz"
LEVELS = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,
          550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]
def load_channel_index():
    names=[l.strip() for l in open(f"{ERA5_DIR}/feature_names.txt")]
    idx={n:i for i,n in enumerate(names)}
    qi=np.array([idx[f"specific_humidity_{L}hPa"] for L in LEVELS])
    ui=np.array([idx[f"u_component_of_wind_{L}hPa"] for L in LEVELS])
    vi=np.array([idx[f"v_component_of_wind_{L}hPa"] for L in LEVELS])
    return idx, np.array(LEVELS,float), qi, ui, vi, idx["latitude"], idx["longitude"]
def region_node_setup():
    idx,_,_,_,_,lat_i,lon_i = load_channel_index()
    arr=np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat=arr[:,lat_i]; nlon=arr[:,lon_i]; grid_lon=np.arange(0,360,0.25)
    d=np.load(NPZ); setup={}
    for r,cfg in REGIONS.items():
        inlat=(nlat>=cfg["lat"][0])&(nlat<=cfg["lat"][1]); inlon=np.zeros_like(inlat)
        for x,y in cfg["lon"]: inlon|=(nlon>=x)&(nlon<=y)
        nodes=np.where(inlat&inlon)[0]
        rlat=d[f"{r}__lat"]
        rlon=np.concatenate([grid_lon[(grid_lon>=x)&(grid_lon<=y)] for x,y in cfg["lon"]])
        li=np.abs(nlat[nodes][:,None]-rlat[None,:]).argmin(1)
        ji=np.abs(nlon[nodes][:,None]-rlon[None,:]).argmin(1)
        setup[r]=dict(nodes=nodes, li=li, ji=ji)
    return setup
def node_ivt(arr, qi, ui, vi, levels):
    return ivt(arr[:,qi], arr[:,ui], arr[:,vi], levels)

def max_ivt_over_ar(arr, region_setup, mask_t, qi, ui, vi, levels):
    """Max IVT over AR-labeled mesh nodes in a region. NaN if no AR nodes."""
    s = region_setup; rn = arr[s["nodes"]]
    iv = ivt(rn[:, qi], rn[:, ui], rn[:, vi], levels)
    arn = mask_t[s["li"], s["ji"]] > 0
    return float(iv[arn].max()) if arn.any() else float("nan")
