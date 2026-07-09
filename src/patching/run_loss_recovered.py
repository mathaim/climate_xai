"""Loss recovered: baseline vs recon-splice vs zero-ablation forwards per event and layer.
recovered = 1 - MSE(sae,base)/MSE(zero,base), on global IVT and T2M fields."""
import os, glob, numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.analysis.ar_intensity.ivt import ivt
LR = "/scratch/euh7ys/climate_xai/patching/lossrec"
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L8_sae.npz"
def fields(pred):
    q = pred["specific_humidity"]; u = pred["u_component_of_wind"]; v = pred["v_component_of_wind"]
    lv = np.asarray(q["level"].values, float); L = len(lv)
    iv = ivt(q.values[0,0].reshape(L,-1).T, u.values[0,0].reshape(L,-1).T, v.values[0,0].reshape(L,-1).T, lv)
    return iv, pred["2m_temperature"].values[0,0].ravel()
def main():
    S = P.setup(NPZ)
    tags = sorted({os.path.basename(f).rsplit("_",1)[-1].replace(".npy","") for f in glob.glob(f"{LR}/drecon_*.npy")})
    print("events:", tags, flush=True)
    for tag in tags:
        T = tag[:13] + ":" + tag[14:]          # 2017-08-21T18-00 -> 2017-08-21T18:00
        inp, tar, frc = P.build_inputs(T, S)
        base = P.run_one(P.make_field_forward(jnp.zeros((40962,1,512), jnp.float32), S, 8), inp, tar, frc)
        base_iv, base_t2 = fields(base)
        for layer in (0, 8, 15):
            fz = f"{LR}/dzero_L{layer}_{tag}.npy"
            if not os.path.exists(fz): print(f"missing {fz}", flush=True); continue
            dz = np.load(fz).astype(np.float32)[:, None, :]
            ziv, zt2 = fields(P.run_one(P.make_field_forward(jnp.asarray(dz), S, layer), inp, tar, frc))
            mz_iv = float(np.nanmean((ziv-base_iv)**2)); mz_t2 = float(np.nanmean((zt2-base_t2)**2))
            print(f"{tag} L{layer} zero-abl: dIVT-MSE {mz_iv:.2f}  dT2M-MSE {mz_t2:.4f}", flush=True)
            for arch in ("matry", "plain"):
                fr = f"{LR}/drecon_{arch}_L{layer}_{tag}.npy"
                if not os.path.exists(fr): continue
                dr = np.load(fr).astype(np.float32)[:, None, :]
                riv, rt2 = fields(P.run_one(P.make_field_forward(jnp.asarray(dr), S, layer), inp, tar, frc))
                r_iv = 1 - float(np.nanmean((riv-base_iv)**2))/max(mz_iv, 1e-9)
                r_t2 = 1 - float(np.nanmean((rt2-base_t2)**2))/max(mz_t2, 1e-9)
                print(f"{tag} L{layer} {arch}: recovered IVT {r_iv:.3f}  T2M {r_t2:.3f}", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__":
    main()
