"""Firing fingerprint: for parent 99 and two children, show region / season / IVT of their
top firing events. Parent broad; children peaked -> the fan-out, via event conditions."""
import numpy as np
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; REG = list(REGIONS)
CONCEPTS = [(99,"Concept 99\n(general intensity)"),(3153,"Child 3153\n(extreme E. Australia)"),(3483,"Child 3483\n(N. Hemisphere)")]
COLORS = {"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
NTOP = 300
def main():
    store = {c:{"act":[],"reg":[],"ivt":[],"mon":[]} for c,_ in CONCEPTS}
    for r in REG:
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); ivt = t["ivt"].astype(float); ok = np.isfinite(ivt)
        ivt = ivt[ok]; mon = t["month"][ok]
        for c,_ in CONCEPTS:
            ca = t["A_max"][ok, c].astype(float)
            store[c]["act"].append(ca); store[c]["reg"].append(np.full(len(ca), r))
            store[c]["ivt"].append(ivt); store[c]["mon"].append(mon)
        del t
    fig, axes = plt.subplots(len(CONCEPTS), 3, figsize=(15, 10))
    for ri,(c,title) in enumerate(CONCEPTS):
        g = {k: np.concatenate(v) for k,v in store[c].items()}
        top = np.argsort(g["act"])[::-1][:NTOP]
        rc = Counter(g["reg"][top])
        axes[ri,0].bar(range(4),[rc.get(k,0) for k in REG],color=[COLORS[k] for k in REG])
        axes[ri,0].set_xticks(range(4)); axes[ri,0].set_xticklabels([k.replace("_","\n") for k in REG],fontsize=7)
        axes[ri,0].set_ylabel(title,fontsize=9)
        mc = Counter(g["mon"][top])
        axes[ri,1].bar(range(1,13),[mc.get(m,0) for m in range(1,13)],color="#555")
        axes[ri,1].set_xticks(range(1,13)); axes[ri,1].set_xlabel("month" if ri==2 else "")
        axes[ri,2].hist(g["ivt"][top],bins=20,color="#2b6cb0")
        axes[ri,2].axvline(250,color="k",ls="--",lw=.8); axes[ri,2].set_xlabel("IVT" if ri==2 else "")
        if ri==0:
            axes[0,0].set_title("region of firing"); axes[0,1].set_title("season of firing"); axes[0,2].set_title("IVT at firing")
    fig.suptitle("Firing fingerprints: parent 99 and its specialized children",y=1.0)
    fig.tight_layout(); fig.savefig(f"{TRACK}/firing_fingerprint.png",dpi=150,bbox_inches="tight")
    print("saved",f"{TRACK}/firing_fingerprint.png")
if __name__=="__main__": main()
