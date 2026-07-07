"""Best vs runner-up L8->L15 Jaccard per concept. Geographic children have a dominant
co-located counterpart; AR children and broad cores do not. Reads cached cross_layer npz."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/scratch/euh7ys/climate_xai/concept_ivt"
def top2(fn, cc):
    d = np.load(f"{D}/{fn}"); k = list(d["wsa"]).index(cc); r = np.sort(d["jac"][k])[::-1]; return r[0], r[1]
rows = [("3481", "geo", *top2("cross_layer.npz", 3481)), ("3948", "geo", *top2("cross_layer.npz", 3948)),
        ("3675", "geo", *top2("cross_layer.npz", 3675)), ("99", "ar", *top2("cross_layer_99.npz", 99)),
        ("1454", "ar", *top2("cross_layer_99.npz", 1454)), ("3392", "ar", *top2("cross_layer_99.npz", 3392)),
        ("2722", "ar", *top2("cross_layer_99.npz", 2722))]
x = np.arange(len(rows)); w = 0.38
cols = ["#2980b9" if r[1] == "geo" else "#c0392b" for r in rows]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - w/2, [r[2] for r in rows], w, color=cols, label="best L15 match")
ax.bar(x + w/2, [r[3] for r in rows], w, color=cols, alpha=0.4, label="2nd-best match")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel("L8$\\to$L15 firing Jaccard")
ax.axvspan(-0.5, 2.5, color="#2980b9", alpha=0.05); ax.axvspan(2.5, 6.5, color="#c0392b", alpha=0.05)
ax.text(1, ax.get_ylim()[1]*0.92, "geographic children", ha="center", color="#2980b9", fontsize=9)
ax.text(4.5, ax.get_ylim()[1]*0.92, "AR-intensity family", ha="center", color="#c0392b", fontsize=9)
ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/xlayer_jaccard.png", dpi=170, bbox_inches="tight"); print("saved xlayer_jaccard.png")
