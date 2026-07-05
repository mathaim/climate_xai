"""Select 3 peak-IVT timestamps (spread across the record) and encode IVT/transport + concept-1592 fields."""
import numpy as np, pandas as pd, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode, act_file
from src.analysis.ar_intensity.regions import index_to_datetime
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
from src.analysis.ar_intensity.ivt import layer_thickness_pa
R, C, SAE, G = "W_N_America", 1592, "plain_L8", 9.81
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; OUT = f"{TRACK}/snapshots_1592.npz"
conv = lambda x: x - 360 if x > 180 else x
def pick_peaks():
    d = np.load(f"{TRACK}/track_pool_{R}.npz"); A = d["A_max"][:, C].astype(float); ivt = d["ivt"].astype(float); ti = d["tindex"]
    dt = np.array([index_to_datetime(int(t)) for t in ti]); o = np.argsort(dt); dt, A, ivt = dt[o], A[o], ivt[o]
    lo, hi = DT.datetime(1985, 1, 1), DT.datetime(2013, 12, 31); m = (dt >= lo) & (dt <= hi); dt, A, ivt = dt[m], A[m], ivt[m]
    di = pd.DatetimeIndex(dt); env = pd.Series(ivt, index=di).resample("2MS").mean()
    picks = []
    for th in np.array_split(np.arange(len(env)), 3):
        wi = env.iloc[th].idxmax(); w0, w1 = wi, wi + pd.DateOffset(months=2); idx = np.where((di >= w0) & (di < w1))[0]
        ni = (ivt[idx] - ivt[idx].min()) / (np.ptp(ivt[idx]) + 1e-9); na = (A[idx] - A[idx].min()) / (np.ptp(A[idx]) + 1e-9)
        picks.append(dt[idx[np.argmax(ni * na)]])
    return picks
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index(); dp = layer_thickness_pa(levels)
    m, c, fmin, frng = load_sae(SAE, dev); picks = pick_peaks()
    print("PEAK TIMESTAMPS:", [p.strftime("%Y-%m-%dT%H:%M") for p in picks], flush=True)
    save = {"labels": np.array([p.strftime("%Y-%m-%d %Hz") for p in picks])}
    for i, dt in enumerate(picks):
        era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        if i == 0:
            save["nlat"] = era[:, lat_i]; save["nlon"] = np.array([conv(x) for x in era[:, lon_i]])
        qu = (era[:, qi] * era[:, ui] * dp[None, :]).sum(1) / G; qv = (era[:, qi] * era[:, vi] * dp[None, :]).sum(1) / G
        a = np.load(act_file(c, dt), mmap_mode="r"); xr = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        with torch.no_grad(): val = encode(m, c["arch"], torch.from_numpy(xr).to(dev)).cpu().numpy()[:, C]
        save[f"mag{i}"] = np.sqrt(qu**2 + qv**2); save[f"qu{i}"] = qu; save[f"qv{i}"] = qv; save[f"val{i}"] = val
        print("encoded", dt, flush=True)
    np.savez(OUT, **save); print("saved", OUT, flush=True)
if __name__ == "__main__":
    main()
