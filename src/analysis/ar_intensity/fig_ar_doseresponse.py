"""Result B: region firing vs AR max_ivt. 99-family (AR-intensity) rises; 1829/3481 flat (sub-AR)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/sae_features"
meta = pd.read_parquet(f"{D}/matry_L8_meta.parquet"); m = (meta["region"] == "W_S_America").values
iv = meta[m]["max_ivt"].values.astype(float); F = np.asarray(np.load(f"{D}/matry_L8_features_region_binary.npy", mmap_mode="r")[m])
bins = np.array([0, 150, 250, 350, 450, 600, 800, 1300]); ctr = (bins[:-1] + bins[1:]) / 2
bi = np.clip(np.digitize(iv, bins) - 1, 0, len(ctr) - 1)
curve = lambda c: np.array([F[bi == b, c].mean() if (bi == b).any() else np.nan for b in range(len(ctr))])
fig, ax = plt.subplots(figsize=(8, 5))
for c, col, lab in [(99, "#c0392b", "99  global AR-intensity"), (1454, "#e67e22", "1454  AR child"),
                    (3392, "#e8b000", "3392  AR child"), (1829, "#2980b9", "1829  (sub-AR, nesting parent)"),
                    (3481, "#27ae60", "3481  (sub-AR, nesting child)")]:
    ax.plot(ctr, curve(c), "-o", ms=5, color=col, label=lab)
ax.axvspan(0, 319, alpha=0.08, color="gray"); ax.axvline(319, color="0.5", ls="--")
ax.text(150, ax.get_ylim()[1] * 0.92, "sub-AR (< p50)", fontsize=9, color="0.4")
ax.set_xlabel("AR intensity, region max IVT (kg m$^{-1}$ s$^{-1}$)"); ax.set_ylabel("region firing (mean active nodes)")
ax.set_title("AR-intensity response: 99 family scales with AR intensity; 1829/3481 are flat (sub-AR)")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig("/scratch/euh7ys/climate_xai/plots/ar_intensity_doseresponse.png", dpi=170, bbox_inches="tight"); print("saved dose-response")
