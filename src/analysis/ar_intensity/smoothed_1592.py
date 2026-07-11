"""1592 robustness: correlation of 2-month-mean regional firing vs 2-month-mean AR intensity,
W_N_America, full 1979-2017 record, from cached pipeline features."""
import numpy as np, pandas as pd
D = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
meta = pd.read_parquet(f"{D}/plain_L8_meta.parquet")
F = np.load(f"{D}/plain_L8_features_region_binary.npy", mmap_mode="r")
from src.analysis.ar_intensity.regions import index_to_datetime
m = (meta["region"] == "W_N_America").values & np.isfinite(meta["max_ivt"].values)
df = pd.DataFrame({"t": [index_to_datetime(int(i)) for i in meta["time_index"].values[m]],
                   "f1592": np.asarray(F[m][:, 1592], float),
                   "ivt": meta["max_ivt"].values[m]}).set_index("t").sort_index()
for w in ("2M", "1M"):
    r = df.resample(w).mean().dropna()
    c = r["f1592"].corr(r["ivt"])
    print(f"{w}-mean correlation, W_N_America, 1979-2017 (n={len(r)}): r = {c:.3f}", flush=True)
print(f"event-level reference: r = {np.corrcoef(df['f1592'], df['ivt'])[0,1]:.3f}", flush=True)
