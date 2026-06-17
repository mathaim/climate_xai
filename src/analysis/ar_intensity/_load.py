import numpy as np, pandas as pd
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
def load(name):
    F=np.load(f"{OUTDIR}/{name}_features.npy")
    F=F.reshape(F.shape[0], -1)          # L8/L15 come in as (N,1,4096); collapse to (N,4096)
    md=pd.read_parquet(f"{OUTDIR}/{name}_meta.parquet").reset_index(drop=True)
    assert len(md)==F.shape[0], f"{name}: md {len(md)} != F {F.shape[0]}"
    keep=md["intensity_bin"].notna().to_numpy()
    if (~keep).sum(): print(f"  [{name}] dropped {int((~keep).sum())} null-bin rows")
    return F[keep], md.loc[keep].reset_index(drop=True)
