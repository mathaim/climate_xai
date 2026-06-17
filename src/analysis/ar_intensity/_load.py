import numpy as np, pandas as pd
OUTDIR="/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
def load(name):
    F=np.load(f"{OUTDIR}/{name}_features.npy")
    md=pd.read_parquet(f"{OUTDIR}/{name}_meta.parquet").reset_index(drop=True)
    assert len(md)==F.shape[0], f"{name}: md {len(md)} != F {F.shape[0]}"
    keep=md["intensity_bin"].notna().to_numpy()
    if (~keep).sum(): print(f"  [{name}] dropped {int((~keep).sum())} null-bin rows")
    return F[keep], md.loc[keep].reset_index(drop=True)
