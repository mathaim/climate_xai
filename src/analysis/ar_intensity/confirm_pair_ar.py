"""Confirm 1829/3481 (strong-AR) vs 340 (non-AR) against the aligned AR ground truth. Split W_S_America
timesteps into no-AR / weak-AR / strong-AR (pipeline qualifies + max_ivt), measure each concept's PEAK
activation in the box (extent-free). Base env, SLURM."""
import os, glob, numpy as np, pandas as pd, torch, datetime as DT
from src.analysis.ar_intensity.sae_features import load_sae, encode, SAES
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index
from src.analysis.ar_intensity.regions import REGIONS
PIPE = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline"
REG = "W_S_America"; CONCEPTS = [1829, 3481, 340]; CORE = (-47.0, 286.0)
def actfile(c, dt): return f"{c['act']}/layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t{dt.strftime('%Y-%m-%dT%H-%M')}.npy"
def main():
    cov = pd.read_parquet(f"{PIPE}/regional_coverage.parquet"); cov = cov[cov.region == REG]
    ivf = pd.read_parquet(f"{PIPE}/ar_intensity_full.parquet"); ivf = ivf[ivf.region == REG][["time_index", "max_ivt"]]
    df = cov.merge(ivf, on="time_index")
    p50 = df.loc[df.qualifies, "max_ivt"].median()
    df["cat"] = np.where(~df.qualifies, "no_AR", np.where(df.max_ivt < p50, "weak_AR", "strong_AR"))
    print("category counts:", df["cat"].value_counts().to_dict(), "| strong/weak split max_ivt=%.0f" % p50, flush=True)
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    era0 = np.load(sorted(glob.glob(f"{c['act']}/../../../climate_xai/data/era5/era5_inputs_*.npy"))[0]) if False else None
    from src.analysis.ar_intensity.ivt_pipeline import ERA5_DIR
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0]); nlat = era0[:, lat_i]; nlon = era0[:, lon_i]
    la = REGIONS[REG]["lat"]; lo = REGIONS[REG]["lon"][0]
    box = (nlat >= la[0]) & (nlat <= la[1]) & (nlon >= lo[0]) & (nlon <= lo[1])
    core = int(np.argmin((nlat - CORE[0])**2 + np.minimum(np.abs(nlon - CORE[1]), 360 - np.abs(nlon - CORE[1]))**2))
    rng = np.random.default_rng(0); res = {cc: {cat: [] for cat in ["no_AR", "weak_AR", "strong_AR"]} for cc in CONCEPTS}
    corecc = {cc: {cat: [] for cat in ["no_AR", "weak_AR", "strong_AR"]} for cc in CONCEPTS}
    for cat in ["no_AR", "weak_AR", "strong_AR"]:
        sub = df[df.cat == cat]; sub = sub.iloc[rng.choice(len(sub), min(120, len(sub)), replace=False)]
        for _, row in sub.iterrows():
            dt = pd.Timestamp(row["datetime"]).to_pydatetime(); f = actfile(c, dt)
            if not os.path.exists(f): continue
            a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
            if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
            with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
            for cc in CONCEPTS:
                res[cc][cat].append(float(acts[box, cc].max())); corecc[cc][cat].append(float(acts[core, cc]))
    print(f"\n{'concept':>7}  {'metric':>10}  {'no_AR':>8}{'weak_AR':>9}{'strong_AR':>11}")
    for cc in CONCEPTS:
        for lab, dd in [("box_peak", res[cc]), ("@47S core", corecc[cc])]:
            print(f"{cc:>7}  {lab:>10}  " + "".join(f"{np.mean(dd[cat]):>9.2f}" for cat in ["no_AR", "weak_AR", "strong_AR"]))
if __name__ == "__main__":
    main()
