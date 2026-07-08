"""Unified dose-response, all 8 concepts: normalized P(concept fires | local IVT bin), a
basin-independent x-axis. Geographic concepts peak at sub-AR IVT; the AR core + children rise
with IVT. Reads L8 activations + ERA5 (for per-node IVT)."""
import os, glob, numpy as np, torch, datetime as DT, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
N = 400; THRESH = 0.1
CC = [(340,"#7f8c8d","-","340  coastal (parent)"), (3481,"#2980b9","-","3481  coastal ~47S"),
      (3948,"#16a085","-","3948  coastal ~38S"),   (3675,"#6b8e23","-","3675  coastal ~35S"),
      (99,"#c0392b","--","99  AR core"),            (1454,"#8e44ad","--","1454  AR child"),
      (3392,"#e67e22","--","3392  AR child"),       (2722,"#d81b60","--","2722  AR child")]
bins = np.array([0, 100, 200, 300, 400, 550, 750, 1100]); ctr = (bins[:-1] + bins[1:]) / 2
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m, c, fmin, frng = load_sae("matry_L8", "cpu")
    files = sorted(glob.glob(f"{c['act']}/layer0008_*.npy")); rng = np.random.default_rng(0)
    sel = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    nb = len(ctr); fire = np.zeros((len(CC), nb)); tot = np.zeros(nb)
    for f in sel:
        dt = pdt(os.path.basename(f))
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels); bi = np.clip(np.digitize(iv, bins) - 1, 0, nb - 1)
        tot += np.bincount(bi, minlength=nb)
        a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1)
        if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
        with torch.no_grad(): acts = encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
        for k, (cc, *_ ) in enumerate(CC):
            fb = bi[acts[:, cc] > THRESH]; fire[k] += np.bincount(fb, minlength=nb)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k, (cc, col, ls, lab) in enumerate(CC):
        r = fire[k] / np.maximum(tot, 1); r = r / np.nanmax(r) if np.nanmax(r) > 0 else r
        ax.plot(ctr, r, ls, marker="o", ms=4, color=col, label=lab)
    ax.axvspan(0, 250, alpha=0.06, color="gray"); ax.axvline(250, color="0.5", ls=":")
    ax.text(80, 0.95, "sub-AR", fontsize=9, color="0.4")
    ax.set_xlabel("local IVT at firing node (kg/m/s)"); ax.set_ylabel("firing likelihood (normalized)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3); fig.tight_layout()
    ax.set_ylim(0, 1.08)
    fig.savefig("/scratch/euh7ys/climate_xai/plots/ar_intensity_doseresponse.png", dpi=170, bbox_inches="tight")
    print("saved ar_intensity_doseresponse.png (all 8)")
if __name__ == "__main__":
    main()
