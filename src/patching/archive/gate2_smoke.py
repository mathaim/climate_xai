"""Smoke test (unpatched prediction) + Gate 2 (injector at alpha=0 reproduces it)."""
import numpy as np, jax.numpy as jnp
from src.patching import patch_predict as P
from src.patching.sae_to_jax import latent_dim
def main():
    T = "2021-11-14T06:00"; S = P.setup()
    print("=== smoke: unpatched forward ===", flush=True)
    predN = P.predict(T, P.make_forward(None, S), S)
    print("vars:", list(predN.data_vars)[:8]); print("q shape:", tuple(predN['specific_humidity'].shape), flush=True)
    print("=== gate 2: alpha=0 identity ===", flush=True)
    pred0 = P.predict(T, P.make_forward(jnp.zeros(latent_dim()), S), S)
    d = float(np.nanmax(np.abs(predN['specific_humidity'].values - pred0['specific_humidity'].values)))
    print("max|alpha0 - unpatched| (specific_humidity) =", d)
    assert d < 1e-3, f"alpha=0 NOT identity (d={d}) -> injector mis-wired"
    print("GATE 2 PASS")
if __name__ == "__main__":
    main()
