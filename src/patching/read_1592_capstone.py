"""Encode plain-L15 captures; read 2251 (functional successor), 1226/3605 (overlap picks),
3163 (regional champion), vs 20-random floor."""
import os, numpy as np, torch
from src.analysis.ar_intensity.sae_features import load_sae, encode
PATCH = "/scratch/euh7ys/climate_xai/patching"; T = [2251, 1226, 3605, 3163]
def enc(tag, m, c):
    fp = f"{PATCH}/l15p_cap_{tag}.npy"
    if not os.path.exists(fp): return None
    x = torch.from_numpy(np.load(fp).astype(np.float32).reshape(-1, 512))
    with torch.no_grad(): return encode(m, c["arch"], x).cpu().numpy()
def main():
    m, c, _, _ = load_sae("plain_L15", "cpu")
    rng = np.random.default_rng(0); rand = rng.choice(4096, 20, replace=False)
    for base_tag, mod_tag in [("ar_base", "ar_clamp1592"), ("clear_base", "clear_add1592")]:
        B, X = enc(base_tag, m, c), enc(mod_tag, m, c)
        if B is None or X is None: print(f"missing {base_tag}/{mod_tag}"); continue
        d = X - B; floor = np.median([np.abs(d[:, j]).sum() for j in rand])
        row = "  ".join(f"{t}:{d[:,t].sum():+.1f}({np.abs(d[:,t]).sum()/max(floor,1e-9):.0f}x)" for t in T)
        print(f"{mod_tag}: floor {floor:.2f} | " + row, flush=True)
if __name__ == "__main__":
    main()
