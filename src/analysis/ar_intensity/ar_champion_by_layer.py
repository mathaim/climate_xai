"""Apply the L8 AR-relevance criterion independently at every (sae, layer): corr(region firing,
max_ivt) over AR events, averaged across the 4 regions. Reports top-10 concepts per combo, their
matryoshka prefix group, and where the L8->L15 Jaccard matches (111 etc.) rank. Precomputed
pipeline features only (no encoding)."""
import numpy as np, pandas as pd
D = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
GRP = lambda c: "G0" if c<256 else "G1" if c<512 else "G2" if c<1024 else "G3" if c<2048 else "G4"
CHECK = {"matry_L15": [111, 3160, 3392, 1980, 864], "matry_L8": [99, 340, 3481, 176, 369, 664]}
for sae in ["matry_L0","matry_L8","matry_L15","plain_L0","plain_L8","plain_L15"]:
    try:
        meta = pd.read_parquet(f"{D}/{sae}_meta.parquet")
        F = np.load(f"{D}/{sae}_features_region_binary.npy", mmap_mode="r")
    except FileNotFoundError:
        print(f"{sae}: features missing, skipped"); continue
    cs = []
    for reg in meta["region"].unique():
        iv_all = meta["max_ivt"].values.astype(float)
        m = (meta["region"]==reg).values & np.isfinite(iv_all)
        iv = iv_all[m]
        if m.sum() < 100: print(f"   {reg}: only {m.sum()} finite rows, skipped"); continue
        X = np.asarray(F[m], dtype=np.float64); Xz = X - X.mean(0); ivz = iv - iv.mean()
        den = np.sqrt((Xz**2).sum(0)*(ivz**2).sum())
        c = np.where(den > 0, (Xz*ivz[:,None]).sum(0)/np.maximum(den,1e-12), 0.0)
        cs.append(c)
        nbad = int((~np.isfinite(iv_all[(meta["region"]==reg).values])).sum())
        if nbad: print(f"   {reg}: dropped {nbad} non-finite max_ivt rows")
    c = np.nan_to_num(np.mean(cs, 0)); top = np.argsort(-c)[:10]
    tag = (lambda x: f"{x}({GRP(x)})") if sae.startswith("matry") else str
    print(f"\n{sae}: max corr {c[top[0]]:+.3f}   top10: " + ", ".join(f"{tag(t)} {c[t]:+.2f}" for t in top))
    for cc in CHECK.get(sae, []):
        print(f"   check {tag(cc)}: corr {c[cc]:+.3f}  rank {int((c>c[cc]).sum())+1}/4096")
