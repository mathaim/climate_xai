"""Keystone figure: peak |r| vs processor layer, Plain vs Matryoshka, for intensity (vs IVT)
and ENSO (vs ONI). Intensity diverges at L15 (Matryoshka sharpens); ENSO is flat."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LAYERS = [0, 8, 15]
INT = {"Plain":{0:[.66,.60,.52,.57],8:[.66,.56,.55,.54],15:[.68,.53,.57,.51]},
       "Matryoshka":{0:[.67,.52,.58,.67],8:[.69,.53,.53,.55],15:[.76,.62,.60,.56]}}
ENSO= {"Plain":{0:[.85,.60,.75,.76],8:[.83,.64,.74,.78],15:[.79,.57,.81,.81]},
       "Matryoshka":{0:[.81,.66,.76,.79],8:[.77,.64,.74,.83],15:[.78,.66,.76,.85]}}
COL = {"Plain":"#c0392b","Matryoshka":"#2b6cb0"}
plt.rcParams.update({"font.size":13,"axes.titlesize":15})
def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax,(data,title) in zip(axes,[(INT,"Intensity  (peak |r| with IVT)"),(ENSO,"ENSO  (peak |r| with ONI)")]):
        for arch in ["Plain","Matryoshka"]:
            for L in LAYERS:
                ax.scatter([L]*4, data[arch][L], color=COL[arch], alpha=.22, s=22, zorder=2)
            means = [float(np.mean(data[arch][L])) for L in LAYERS]
            ax.plot(LAYERS, means, "-o", color=COL[arch], lw=2.4, ms=8, label=arch, zorder=3)
        ax.set_title(title); ax.set_xlabel("Processor layer"); ax.set_xticks(LAYERS); ax.grid(alpha=.3)
    axes[0].set_ylabel("Peak |r|  (faint = per region, bold = mean)"); axes[0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/refinement_by_layer.png", dpi=200, bbox_inches="tight")
    print("saved /scratch/euh7ys/climate_xai/plots/refinement_by_layer.png")
if __name__ == "__main__":
    main()
