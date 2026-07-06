"""Gate 1: my JAX weight mapping must reproduce the real PlainSAE active features."""
import glob, numpy as np, torch
from src.patching.sae_to_jax import load_l15_weights
from src.analysis.ar_intensity.sae_features import load_sae, encode
DIR = "/project/AikyamLab/madelyn/GraphCast/activations/Layer15"
def test_weight_mapping_matches_plain_sae():
    enc_w, dec_w, b_pre, k = load_l15_weights()
    cand = sorted(glob.glob(f"{DIR}/layer0015_*.npy")) or [f for f in sorted(glob.glob(f"{DIR}/*.npy")) if "feature" not in f]
    assert cand, f"no L15 activation files in {DIR}"
    a = np.load(cand[0]).astype(np.float32).reshape(-1, 512)[:512]
    # ground truth: the actual PlainSAE encode (relu+topk) that defines the concepts
    m, c, _, _ = load_sae("plain_L15", "cpu")
    with torch.no_grad():
        code_t = encode(m, "plain", torch.from_numpy(a)).numpy()
    set_t = [set(np.nonzero(r)[0].tolist()) for r in code_t]
    # numpy replica using my JAX param mapping (same formula SAEInjector uses)
    xn = a - a.mean(1, keepdims=True); xn = xn / np.clip(np.linalg.norm(xn, axis=1, keepdims=True), 1e-6, None)
    cp = np.maximum((xn - b_pre) @ enc_w, 0.0)
    idx_n = np.argsort(cp, 1)[:, -k:]
    set_n = [set(r.tolist()) for r in idx_n]
    overlap = float(np.mean([len(s & n) / k for s, n in zip(set_t, set_n)]))
    print("active-feature overlap:", round(overlap, 4))
    assert overlap > 0.99, f"weight mapping mismatch (overlap {overlap})"
if __name__ == "__main__":
    test_weight_mapping_matches_plain_sae(); print("GATE 1 PASS")
