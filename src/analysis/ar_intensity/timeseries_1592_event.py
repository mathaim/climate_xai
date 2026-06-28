"""W. N. America concept 1592 activation and region IVT over the 2012-13 wet season,
with the 2012-11-28 06z AR (the climate-map event) marked."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TRACK = "/scratch/euh7ys/climate_xai/concept_ivt"; AR_START = pd.Timestamp("1979-01-01")
CONCEPT = 1592; EVENT = pd.Timestamp("2012-11-28 06:00")
WIN = (pd.Timestamp("2012-10-01"), pd.Timestamp("2013-03-01"))
plt.rcParams.update({"font.size": 12})
def main():
    t = np.load(f"{TRACK}/track_pool_W_N_America.npz")
    a = t["A_max"][:, CONCEPT].astype(float); ivt = t["ivt"].astype(float); ti = t["tindex"]
    dates = AR_START + pd.to_timedelta(6*(ti-1), unit="h")
    m = (dates >= WIN[0]) & (dates <= WIN[1]) & np.isfinite(ivt)
    d, aa, iv = dates[m], a[m], ivt[m]
    fig, ax = plt.subplots(figsize=(14, 5))
    l1, = ax.plot(d, iv, color="#2b6cb0", lw=1.1, label="Region max IVT")
    ax.axhline(250, color="#2b6cb0", ls=":", lw=.9)
    ax.set_ylabel("IVT (kg m$^{-1}$ s$^{-1}$)", color="#2b6cb0"); ax.tick_params(axis="y", colors="#2b6cb0")
    ax.set_xlabel("Date  (2012-13 wet season)")
    ax2 = ax.twinx()
    l2, = ax2.plot(d, aa, color="#c0392b", lw=1.1, label="Concept 1592 activation")
    ax2.set_ylabel("Concept 1592 activation", color="#c0392b"); ax2.tick_params(axis="y", colors="#c0392b")
    ax.axvline(EVENT, color="k", ls="--", lw=1.3)
    ax.annotate("2012-11-28 06z\n(spatial-map event)", xy=(EVENT, ax.get_ylim()[1]*0.96),
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))
    ax.legend(handles=[l1, l2, plt.Line2D([],[],color="#2b6cb0",ls=":",label="AR threshold (250)")],
              loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{TRACK}/timeseries_1592_event.png", dpi=180, bbox_inches="tight")
    print("saved", f"{TRACK}/timeseries_1592_event.png")
if __name__ == "__main__":
    main()
