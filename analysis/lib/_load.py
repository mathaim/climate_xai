import numpy as np, pandas as pd
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
def load(name, variant):
    F=np.load(f"{OUTDIR}/{name}_features_{variant}.npy"); F=F.reshape(F.shape[0],-1)
    md=pd.read_parquet(f"{OUTDIR}/{name}_meta.parquet").reset_index(drop=True)
    assert len(md)==F.shape[0], f"{name}/{variant}: md {len(md)} != F {F.shape[0]}"
    keep=md["intensity_bin"].notna().to_numpy()
    return F[keep], md.loc[keep].reset_index(drop=True)
