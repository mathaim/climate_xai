import numpy as np
from src.analysis.ar_intensity.regions import REGIONS
TRACK="/scratch/euh7ys/climate_xai/concept_ivt"; BINS=[256,512,1024,2048]; idx=np.arange(4096); shell=np.digitize(idx,BINS)
rank=lambda x:np.argsort(np.argsort(x)).astype(float); pc=lambda a,b:np.corrcoef(a,b)[0,1]
for SAE,stem in [("matry_L8","track_matry"),("plain_L8","track_pool")]:
    dens=np.zeros(4096); freq=np.zeros(4096); nt=0
    for r in REGIONS:
        d=np.load(f"{TRACK}/{stem}_{r}.npz"); Am=d["A_mean"].astype(np.float64); Ax=d["A_max"]
        dens+=Am.sum(0); freq+=(Ax>0).sum(0); nt+=Am.shape[0]; del d,Am,Ax
    dens/=nt; freq/=nt   # mean activation density; firing frequency (fraction of timesteps active)
    print(f"\n### {SAE}")
    print(f"  corr(activation density, index) = {pc(rank(dens),rank(idx)):+.3f}")
    print(f"  corr(firing frequency,   index) = {pc(rank(freq),rank(idx)):+.3f}")
    print(f"  {'shell':>6} {'n':>5} {'mean density':>13} {'mean freq':>11}")
    for s in range(5):
        m=shell==s; print(f"  {s:>6} {int(m.sum()):>5} {dens[m].mean():>13.4f} {freq[m].mean():>11.4f}")
