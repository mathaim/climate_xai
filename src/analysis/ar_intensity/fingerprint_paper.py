"""Paper figure: firing fingerprints of parent 99 and children 3153/3483 - the region,
season, and IVT of their 300 strongest firing events."""
import numpy as np
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; REGN = list(REGIONS)
ROWS = [(99,  "Concept 99 (parent)\nGeneral intensity"),
        (3153,"Concept 3153 (child)\nExtreme E. Australia"),
        (3483,"Concept 3483 (child)\nNorthern Hemisphere")]
RCOL = {"W_N_America":"#c0392b","W_Europe":"#2b6cb0","W_S_America":"#27ae60","E_Australia":"#8e44ad"}
RSHORT = {"W_N_America":"W.N.Am","W_Europe":"W.Eu","W_S_America":"W.S.Am","E_Australia":"E.Aus"}
MLET = ["J","F","M","A","M","J","J","A","S","O","N","D"]; NTOP = 300
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 11})
def main():
    store = {c: {"act":[],"reg":[],"ivt":[],"mon":[]} for c,_ in ROWS}
    for r in REGN:
        t = np.load(f"{TRACK}/track_matry_{r}.npz"); ivt = t["ivt"].astype(float); ok = np.isfinite(ivt)
        ivt = ivt[ok]; mon = t["month"][ok]
        for c,_ in ROWS:
            ca = t["A_max"][ok, c].astype(float)
            store[c]["act"].append(ca); store[c]["reg"].append(np.full(len(ca), r))
            store[c]["ivt"].append(ivt); store[c]["mon"].append(mon)
        del t
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(15, 10))
    for ri,(c,label) in enumerate(ROWS):
        g = {k: np.concatenate(v) for k,v in store[c].items()}
        top = np.argsort(g["act"])[::-1][:NTOP]
        rc = Counter(g["reg"][top])
        a0 = axes[ri][0]; a0.bar(range(4),[rc.get(k,0) for k in REGN],color=[RCOL[k] for k in REGN])
        a0.set_xticks(range(4)); a0.set_xticklabels([RSHORT[k] for k in REGN], fontsize=9)
        a0.set_ylabel(label, fontsize=10.5)
        mc = Counter(g["mon"][top])
        a1 = axes[ri][1]; a1.bar(range(1,13),[mc.get(m,0) for m in range(1,13)], color="#555")
        a1.set_xticks(range(1,13)); a1.set_xticklabels(MLET, fontsize=9)
        a2 = axes[ri][2]; a2.hist(g["ivt"][top], bins=20, color="#2b6cb0")
        a2.axvline(250, color="k", ls="--", lw=1)
        if ri == 0:
            a2.text(290, a2.get_ylim()[1]*0.95, "AR\nthreshold", fontsize=8, va="top")
            a0.set_title("Region of firing"); a1.set_title("Season of firing"); a2.set_title("IVT at firing")
        if ri == len(ROWS)-1:
            a0.set_xlabel("Region"); a1.set_xlabel("Month"); a2.set_xlabel("IVT (kg m$^{-1}$ s$^{-1}$)")
    fig.supylabel("Count among 300 strongest firing events", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{TRACK}/firing_fingerprint_paper.png", dpi=200, bbox_inches="tight")
    print("saved", f"{TRACK}/firing_fingerprint_paper.png")
if __name__ == "__main__":
    main()
