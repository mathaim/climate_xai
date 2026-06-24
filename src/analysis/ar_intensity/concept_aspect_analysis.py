"""Top-3 IVT-correlated concepts per region; high/low-corr timesteps by season x regime,
over the full record (dry + AR). Reports tracking vs max_ivt (core) and mean_ivt (broad)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.regions import REGIONS
from src.analysis.ar_intensity import concept_ivt_core as C
OUT = "/scratch/euh7ys/climate_xai/concept_ivt"
REGIMES = ["no_ar", "weak", "moderate", "intense"]
AR_REGIMES = ["weak", "moderate", "intense"]

def analyze_region(r):
    d = np.load(f"{OUT}/track_{r}.npz", allow_pickle=True)
    A, ivt, ivtm, month = d["A"].astype(float), d["ivt"].astype(float), d["ivt_mean"].astype(float), d["month"]
    ok = np.isfinite(ivt) & np.isfinite(ivtm)
    A, ivt, ivtm, month = A[ok], ivt[ok], ivtm[ok], month[ok]
    idx, r_all = C.select_top_concepts(A, ivt, k=3)            # selection on max_ivt (primary)
    r_mean = C.pearson_cols(A, ivtm)                           # also correlate vs mean_ivt
    season = C.season_label(month, C.HEMISPHERE[r]); regime = C.ivt_regime(ivt)
    zi = C.zscore(ivt)
    sel_rows, prof_rows = [], []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for rank, ci in enumerate(idx):
        za = C.zscore(A[:, ci]); gap = C.pointwise_gap(za, zi); lab = C.classify_corr(gap)
        sel_rows.append({"region": r, "rank": rank + 1, "concept": int(ci),
                         "pearson_max": float(r_all[ci]), "pearson_mean": float(r_mean[ci]),
                         "tracks": "core" if r_all[ci] >= r_mean[ci] else "broad"})
        for reg in REGIMES:
            mreg = regime == reg
            for cls in ["high_corr", "low_corr"]:
                prof_rows.append({"region": r, "concept": int(ci), "regime": reg, "class": cls,
                                  "wet": int(((lab == cls) & mreg & (season == "wet")).sum()),
                                  "dry": int(((lab == cls) & mreg & (season == "dry")).sum())})
        rate = []
        for reg in AR_REGIMES:
            mreg = regime == reg
            hi = ((lab == "high_corr") & mreg).sum(); lo = ((lab == "low_corr") & mreg).sum()
            rate.append(hi / (hi + lo) if (hi + lo) else np.nan)
        ax = axes[rank]; ax.bar(AR_REGIMES, rate, color=["#9ecae1", "#4292c6", "#08519c"])
        ax.set_ylim(0, 1); ax.axhline(0.5, color="#888", lw=.8, ls="--")
        ax.set_title(f"concept {int(ci)} (r={r_all[ci]:.2f})")
        if rank == 0: ax.set_ylabel("tracking rate\n(high-corr fraction)")
    fig.suptitle(r); fig.tight_layout(); fig.savefig(f"{OUT}/aspect_{r}.png", dpi=160); plt.close(fig)
    return sel_rows, prof_rows

def main():
    sel, prof = [], []
    for r in REGIONS:
        s, p = analyze_region(r); sel += s; prof += p
    pd.DataFrame(sel).to_csv(f"{OUT}/concept_select.csv", index=False)
    pdf = pd.DataFrame(prof); pdf.to_csv(f"{OUT}/aspect_profile.csv", index=False)
    print(pdf.to_string()); print("ANALYSIS DONE")

if __name__ == "__main__":
    main()
