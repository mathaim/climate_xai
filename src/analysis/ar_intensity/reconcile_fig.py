import numpy as np
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"
def cors(a,y):
    a=a.astype(float); ok=np.isfinite(y)&np.isfinite(a); a=a[ok]; y=y[ok]
    p=np.corrcoef(a,y)[0,1]; s=np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(y)))[0,1]
    return p,s
pairs=[(1592,"W_N_America",0.61),(2948,"W_Europe",0.43),(3218,"W_S_America",0.50),(3720,"E_Australia",0.48)]
stems={"plain_L8":"track_pool","matry_L8":"track_matry","plain_L15":"track_plain_L15","matry_L15":"track_matry_L15"}
for cid,region,fig in pairs:
    print(f"\nConcept {cid} vs {region}   (figure full-record r={fig})")
    for name,stem in stems.items():
        try:
            d=np.load(f"{TRACK}/{stem}_{region}.npz"); ivt=d["ivt"].astype(float); A=d["A_max"][:,cid]; del d
            p,s=cors(A,ivt); print(f"   {name:10} Pearson={p:+.3f}  Spearman={s:+.3f}")
        except Exception as e: print(f"   {name:10} -- {type(e).__name__}")
