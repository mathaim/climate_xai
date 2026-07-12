"""Loss recovered over the FULL predicted state: per-channel dimensionless scores saved to
npz; prints mean/median/min per (event, layer, arch)."""
import os, glob, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
LR = "/scratch/euh7ys/climate_xai/patching/lossrec"
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
def channels(pred):
    out = {}
    for v in sorted(pred.data_vars):
        a = pred[v].values
        if a.ndim == 5:
            for li, lev in enumerate(pred[v]["level"].values):
                out[f"{v}_{int(lev)}"] = a[0, 0, li]
        else:
            out[v] = a[0, 0]
    return out
def main():
    S = P.setup(NPZ); store = {}
    tags = sorted({os.path.basename(f).rsplit("_", 1)[-1].replace(".npy","") for f in glob.glob(f"{LR}/drecon_*.npy")})
    names_saved = False
    for tag in tags:
        T = tag[:13] + ":" + tag[14:]
        inp, tar, frc = P.build_inputs(T, S)
        base = channels(P.run_one(P.make_field_forward(jnp.zeros((40962,1,512), jnp.float32), S, 8), inp, tar, frc))
        names = sorted(base)
        for layer in (0, 8, 15):
            dz = np.load(f"{LR}/dzero_L{layer}_{tag}.npy").astype(np.float32)[:, None, :]
            zero = channels(P.run_one(P.make_field_forward(jnp.asarray(dz), S, layer), inp, tar, frc))
            mz = {c: float(np.nanmean((zero[c]-base[c])**2)) for c in names}
            for arch in ("matry", "plain"):
                fr = f"{LR}/drecon_{arch}_L{layer}_{tag}.npy"
                if not os.path.exists(fr): continue
                dr = np.load(fr).astype(np.float32)[:, None, :]
                sp = channels(P.run_one(P.make_field_forward(jnp.asarray(dr), S, layer), inp, tar, frc))
                rec = np.array([1 - float(np.nanmean((sp[c]-base[c])**2))/max(mz[c], 1e-12) for c in names])
                store[f"{arch}_L{layer}_{tag}"] = rec
                w = names[int(np.argmin(rec))]
                print(f"{tag} L{layer} {arch}: mean {rec.mean():.3f} median {np.median(rec):.3f} "
                      f"min {rec.min():.3f} (worst {w}) n_ch {len(rec)}", flush=True)
        if not names_saved:
            store["channel_names"] = np.array(names); names_saved = True
    np.savez(f"{LR}/full_recovered.npz", **store)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
