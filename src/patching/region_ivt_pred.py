"""W_N_America region IVT (max/mean) from a predicted gridded GraphCast state."""
import numpy as np
from src.analysis.ar_intensity.ivt import ivt
LAT = (30, 50); LON_W = (-130, -115)   # W_N_America box (deg)
def region_ivt(pred):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    levels = np.asarray(q["level"].values, float)
    lat = q["lat"].values; lon = ((q["lon"].values + 180) % 360) - 180
    la = (lat >= LAT[0]) & (lat <= LAT[1]); lo = (lon >= LON_W[0]) & (lon <= LON_W[1])
    qs = q.values[0, 0][:, la][:, :, lo]; us = u.values[0, 0][:, la][:, :, lo]; vs = v.values[0, 0][:, la][:, :, lo]
    qf = qs.reshape(len(levels), -1).T; uf = us.reshape(len(levels), -1).T; vf = vs.reshape(len(levels), -1).T
    iv = ivt(qf, uf, vf, levels)
    return float(np.nanmax(iv)), float(np.nanmean(iv))
