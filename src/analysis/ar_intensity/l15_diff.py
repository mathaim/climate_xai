"""Full injection->L15 effect (no node/latent pre-limit): raw activation change per node +
full code change per L15 latent, from the captured baseline/clamp L15 activations."""
import os, glob, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index, ERA5_DIR
PATCH = "/scratch/euh7ys/climate_xai/patching"; conv = lambda x: x - 360 if x > 180 else x
def enc(x, m, c, fmin, frng):
    x = x.astype(np.float32).reshape(-1, 512)
    if fmin is not None: x = (2.0 * (x - fmin) / frng - 1.0).astype(np.float32)
    with torch.no_grad(): return encode(m, c["arch"], torch.from_numpy(x)).cpu().numpy()
def main():
    idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index()
    era0 = np.load(sorted(glob.glob(f"{ERA5_DIR}/era5_inputs_*.npy"))[0])
    nlat = era0[:, lat_i]; nlon = np.array([conv(x) for x in era0[:, lon_i]])
    m, c, fmin, frng = load_sae("matry_L15", "cpu"); base = np.load(f"{PATCH}/l15_cap_baseline.npy")
    Cb = enc(base, m, c, fmin, frng)
    for cc in [3481, 340]:
        fp = f"{PATCH}/l15_cap_clamp_{cc}.npy"
        if not os.path.exists(fp): continue
        clamp = np.load(fp); dx = (clamp - base).reshape(-1, 512); nc = np.linalg.norm(dx, axis=1); pk = int(nc.argmax())
        print(f"\n== clamp {cc} ==")
        print(f"  raw L15 activation change: max |dx|={nc[pk]:.3f} @node {pk} ({nlat[pk]:.0f},{nlon[pk]:.0f}); mean={nc.mean():.5f}; #nodes>0.01={int((nc>0.01).sum())}")
        Cc = enc(clamp, m, c, fmin, frng); dcode = Cc - Cb; lat_tot = np.abs(dcode).sum(0); top = np.argsort(-lat_tot)[:8]
        print("  most-changed L15 latents: " + ", ".join(f"{j}({lat_tot[j]:.1f})" for j in top))
        for tgt in [3160]:
            d = dcode[:, tgt]; p = int(np.abs(d).argmax())
            print(f"  L15 {tgt}: max change {d[p]:+.4f} @node {p} ({nlat[p]:.0f},{nlon[p]:.0f}); total |change|={np.abs(d).sum():.3f}")
if __name__ == "__main__":
    main()
