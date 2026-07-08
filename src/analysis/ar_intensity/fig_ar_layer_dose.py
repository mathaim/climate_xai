"""Functional L8-vs-L15 comparison for the AR family: does each L8 AR concept's L15 match share
its IVT response? Left = L8 concepts P(fire|IVT); right = their L15 matches. A real cross-layer
match keeps the rising AR response; a spurious one is flat. Reads L8+L15 activations + ERA5 IVT."""
import os, glob, numpy as np, torch, datetime as DT, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, node_ivt, ERA5_DIR
N = 350; THRESH = 0.1
PAIRS = [(99,111,"#c0392b"), (1454,864,"#8e44ad"), (3392,3735,"#e67e22"), (2722,4036,"#d81b60")]
L8C = [p[0] for p in PAIRS]; L15C = [p[1] for p in PAIRS]
bins = np.array([0,100,200,300,400,550,750,1100]); ctr = (bins[:-1]+bins[1:])/2; nb = len(ctr)
def pdt(fn): return DT.datetime.strptime(fn.split("_t")[-1].replace(".npy",""), "%Y-%m-%dT%H-%M")
def enc(f, m, c, fmin, frng):
    a = np.load(f, mmap_mode="r"); x = np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0],-1)
    if fmin is not None: x = (2.0*(x-fmin)/frng-1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x).to("cpu")).cpu().numpy()
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    m8,c8,fmin8,frng8 = load_sae("matry_L8","cpu"); m15,c15,fmin15,frng15 = load_sae("matry_L15","cpu")
    dtmap = lambda dd: {pdt(os.path.basename(f)): f for f in glob.glob(f"{dd}/layer*_*.npy")}
    f8,f15 = dtmap(c8["act"]), dtmap(c15["act"]); shared = sorted(set(f8)&set(f15))
    rng = np.random.default_rng(0); sel = [shared[i] for i in rng.choice(len(shared), min(N,len(shared)), replace=False)]
    fire8 = np.zeros((len(PAIRS),nb)); fire15 = np.zeros((len(PAIRS),nb)); tot = np.zeros(nb)
    for dt in sel:
        try: era = np.load(f"{ERA5_DIR}/era5_inputs_{dt.strftime('%Y-%m-%dT%H-%M')}.npy")
        except FileNotFoundError: continue
        iv = node_ivt(era, qi, ui, vi, levels); bi = np.clip(np.digitize(iv, bins)-1, 0, nb-1)
        tot += np.bincount(bi, minlength=nb)
        a8 = enc(f8[dt], m8, c8, fmin8, frng8); a15 = enc(f15[dt], m15, c15, fmin15, frng15)
        for k in range(len(PAIRS)):
            fire8[k]  += np.bincount(bi[a8[:, L8C[k]]  > THRESH], minlength=nb)
            fire15[k] += np.bincount(bi[a15[:, L15C[k]] > THRESH], minlength=nb)
    norm = lambda v: (v/np.maximum(tot,1)) / max(np.nanmax(v/np.maximum(tot,1)), 1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for k,(l8,l15,col) in enumerate(PAIRS):
        ax[0].plot(ctr, norm(fire8[k]),  "-o",  ms=4, color=col, label=f"L8 {l8}")
        ax[1].plot(ctr, norm(fire15[k]), "--o", ms=4, color=col, label=f"L15 {l15}")
    for a, t in zip(ax, ["layer 8 (AR concepts)", "layer 15 (their matches)"]):
        a.axvspan(0,250,alpha=0.06,color="gray"); a.axvline(250,color="0.5",ls=":")
        a.set_xlabel("local IVT at firing node (kg/m/s)"); a.set_title(t); a.legend(fontsize=8); a.grid(alpha=.3)
    ax[0].set_ylabel("firing likelihood (normalized)"); ax[0].set_ylim(0, 1.08)
    fig.tight_layout(); fig.savefig("/scratch/euh7ys/climate_xai/plots/ar_layer_dose.png", dpi=170, bbox_inches="tight")
    print("saved ar_layer_dose.png")
if __name__ == "__main__":
    main()
