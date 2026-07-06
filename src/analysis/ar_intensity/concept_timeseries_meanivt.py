"""Top mean-IVT concept per region (max-pooled): activation vs region MEAN IVT over time,
to see when each concept is maximally activated relative to mean IVT."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity import concept_ivt_core as C
OUT = "/scratch/euh7ys/climate_xai/concept_ivt"
AR_START = pd.Timestamp("1979-01-01 00:00")          # index 1 -> 1979-01-01 00:00 (confirmed)
TOP = {"W_N_America": 1587, "W_Europe": 857, "W_S_America": 3218, "E_Australia": 3218}
YEARS = [2013, 2014, 2015]                            # window for the overlay panel
NPEAK = 100                                           # top activation timesteps

def main():
    fig, axes = plt.subplots(len(REGIONS), 2, figsize=(16, 12),
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, r in enumerate(REGIONS):
        ci = TOP[r]
        d = np.load(f"{OUT}/track_pool_{r}.npz")
        act = d["A_max"][:, ci].astype(float)
        ivtm = d["ivt_mean"].astype(float); tindex = d["tindex"]
        del d
        ok = np.isfinite(ivtm) & np.isfinite(act)
        act, ivtm, tindex = act[ok], ivtm[ok], tindex[ok]
        dates = AR_START + pd.to_timedelta(6 * (tindex - 1), unit="h")
        za, zi = C.zscore(act), C.zscore(ivtm)
        rmi = float(np.corrcoef(act, ivtm)[0, 1])
        order = np.argsort(act)[::-1][:NPEAK]
        pctrank = ivtm.argsort().argsort() / (len(ivtm) - 1)        # IVT percentile per timestep
        peak_pct = float(np.median(pctrank[order]) * 100)
        # overlay over the multi-year window
        yrs = pd.DatetimeIndex(dates).year; w = np.isin(yrs, YEARS)
        ax = axes[row, 0]
        ax.plot(dates[w], zi[w], color="#2b6cb0", lw=.7, label="mean IVT (z)")
        ax.plot(dates[w], za[w], color="#c0392b", lw=.7, label="activation (z)")
        wi = np.where(w)[0]; topw = wi[np.argsort(act[wi])[::-1][:15]]
        ax.scatter(dates[topw], za[topw], color="k", s=16, zorder=5, label="activation peaks")
        ax.set_ylabel("z-score")
        ax.set_title(f"{r}  concept {ci}   r(mean IVT)={rmi:.2f}   "
                     f"peak activation at mean-IVT pct {peak_pct:.0f}")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
        # full-record scatter, peaks highlighted
        ax2 = axes[row, 1]
        ax2.scatter(ivtm, act, s=2, alpha=.12, color="#999")
        ax2.scatter(ivtm[order], act[order], s=12, color="#c0392b")
        ax2.set_xlabel("mean IVT"); ax2.set_ylabel(f"act c{ci}")
    fig.tight_layout(); fig.savefig(f"{OUT}/timeseries_meanivt.png", dpi=150)
    print("saved", f"{OUT}/timeseries_meanivt.png")

if __name__ == "__main__":
    main()
