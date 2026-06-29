"""Load the Plain L15 SAE weights (pre-converted to .npz) into graphcast SAEStaticParams.
Torch-free so it runs in graphcast_sae_env."""
import numpy as np
NPZ = "/scratch/euh7ys/climate_xai/patching/plain_L15_sae.npz"
def load_l15_weights():
    d = np.load(NPZ)
    return (d["enc_w"].astype(np.float32), d["dec_w"].astype(np.float32),
            d["b_pre"].astype(np.float32), int(d["k_active"]))
def load_l15_sae():
    import jax.numpy as jnp
    from graphcast.deep_typed_graph_net import SAEStaticParams
    enc_w, dec_w, b_pre, k = load_l15_weights()
    return SAEStaticParams(enc_w=jnp.asarray(enc_w), dec_w=jnp.asarray(dec_w),
                           b_pre=jnp.asarray(b_pre), k_active=k, unit_norm_decoder=True)
def latent_dim(): return 4096
