"""z(activation) - z(mean IVT) over the FULL 40-yr record, per region's top mean-IVT
concept (max-pooled). Mean-zero by construction; spread ~ sqrt(2(1-r)); structured
negative excursions = saturation at extreme IVT."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity import concept_ivt_core as C
OUT = "/scratch/euh7ys/climate_xai/concept_ivt"
AR_START = pd.Timestamp("1979-01-01 00:00")
TOP = {"W_N_America": 1587, "W_Europe": 857, "W_S_America": 3218, "E_Australia": 3218}

def main():
    fig, axes = plt.subplots(len(REGIONS), 2, figsize=(17, 12), sharey="row",
                             gridspec_kw={"width_ratios": [4, 1]})
    for row, r in enumerate(REGIONS):
        ci = TOP[r]
        d = np.load(f"{OUT}/track_pool_{r}.npz")
        act = d["A_max"][:, ci].astype(float); ivtm = d["ivt_mean"].astype(float); tindex = d["tindex"]
        del d
        ok = np.isfinite(ivtm) & np.isfinite(act)
        act, ivtm, tindex = act[ok], ivtm[ok], tindex[ok]
        dates = AR_START + pd.to_timedelta(6 * (tindex - 1), unit="h")
        diff = C.zscore(act) - C.zscore(ivtm)
        sd = float(diff.std()); frac = float((np.abs(diff) < 1).mean())
        roll = uniform_filter1d(diff, 120, mode="nearest")          # ~30-day smooth
        ax = axes[row, 0]
        ax.plot(dates, diff, lw=.2, alpha=.25, color="#999")
        ax.plot(dates, roll, lw=.9, color="#c0392b", label="30-day mean")
        ax.axhline(0, color="k", lw=.6)
        ax.axhline(sd, color="#2b6cb0", lw=.6, ls="--"); ax.axhline(-sd, color="#2b6cb0", lw=.6, ls="--")
        ax.set_ylabel("z(act) - z(IVT)")
        ax.set_title(f"{r}  concept {ci}   std={sd:.2f}   frac |diff|<1 = {frac:.0%}")
        if row == 0: ax.legend(loc="upper right", fontsize=8)
        ax2 = axes[row, 1]
        ax2.hist(diff, bins=80, color="#888", orientation="horizontal")
        ax2.axhline(0, color="k", lw=.6); ax2.set_xlabel("count")
    fig.tight_layout(); fig.savefig(f"{OUT}/diff_meanivt_40yr.png", dpi=150)
    print("saved", f"{OUT}/diff_meanivt_40yr.png")

if __name__ == "__main__":
    main()
