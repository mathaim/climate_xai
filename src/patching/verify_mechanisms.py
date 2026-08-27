"""Verify the Matryoshka field equals the in-pass edit, and check the sign.
Memory-light: chunked fp32 pass to find firing nodes, fp64 verification on that subset.
Run: python -m src.patching.verify_mechanisms"""
import glob, numpy as np, torch, torch.nn.functional as F
from src.analysis.ar_intensity.sae_features import load_sae, SAES

NAME, CONCEPT, G = "matry_L8", 340, 0.0
m, _, fmin, frng = load_sae(NAME, "cpu")
fmn, frg = torch.tensor(fmin), torch.tensor(frng)
ra = float(m.normalizer.running_avg.detach().cpu().numpy().item())
s = float(np.sqrt(512) / ra)
W_enc, b_enc, W_dec = m.W_enc.detach(), m.b_enc.detach(), m.W_dec.detach()

f = sorted(glob.glob(f"{SAES[NAME]['act']}/layer0008_*.npy"))[0]
a = np.load(f, mmap_mode="r")
x8_all = torch.from_numpy(np.ascontiguousarray(a).astype(np.float32).reshape(a.shape[0], -1))

def encode32(x):                      # fp32, chunk-sized input
    x_mm = 2 * (x - fmn) / frg - 1
    z = F.relu((s * x_mm) @ W_enc + b_enc)
    vals, ixk = torch.topk(z, 32, dim=1)
    mask = torch.zeros_like(z); mask.scatter_(1, ixk, 1.0)
    return z * mask

# pass 1 (fp32, chunked): find nodes where the concept fires
fire_idx = []
with torch.no_grad():
    for i in range(0, x8_all.shape[0], 4096):
        zc_chunk = encode32(x8_all[i:i+4096])[:, CONCEPT]
        fire_idx += (i + torch.nonzero(zc_chunk > 0).squeeze(1)).tolist()
print(f"concept {CONCEPT} fires on {len(fire_idx)} of {x8_all.shape[0]} nodes")
rng = np.random.default_rng(0)
extra = rng.choice(x8_all.shape[0], 500, replace=False).tolist()
sub = sorted(set(fire_idx + extra))

# pass 2 (fp64, subset only): the actual verification
x8 = x8_all[sub].double()
W_enc, b_enc, W_dec, fmn, frg = [t.double() for t in (W_enc, b_enc, W_dec, fmn, frg)]
x_mm = 2 * (x8 - fmn) / frg - 1
z = F.relu((s * x_mm) @ W_enc + b_enc)
vals, ixk = torch.topk(z, 32, dim=1)
mask = torch.zeros_like(z); mask.scatter_(1, ixk, 1.0); z = z * mask
zc = z[:, CONCEPT]

field = (G - 1) * zc[:, None] * W_dec[CONCEPT][None, :] * frg[None, :] / (2 * s)
z_mod = z.clone(); z_mod[:, CONCEPT] = G * zc
dx_raw = (((z_mod - z) @ W_dec) / s) * frg[None, :] / 2
print(f"A vs B  max|diff| = {(field - dx_raw).abs().max():.3e}  (expect ~1e-12: mechanisms equivalent)")

stored_form = -(zc[:, None] * W_dec[CONCEPT][None, :]) * frg[None, :] / (2 * s)
print(f"A vs stored delta_clamp form  max|diff| = {(field - stored_form).abs().max():.3e}  (expect 0 at g=0)")

paper = (1 - G) * zc[:, None] * W_dec[CONCEPT][None, :] * frg[None, :] / (2 * s)
print(f"paper-eq vs code field: max|paper - field| = {(paper - field).abs().max():.3e}, "
      f"max|paper + field| = {(paper + field).abs().max():.3e}")
print("  -> second ~0 means the paper's (1-g) is sign-flipped vs the code")

x_edit = x8 + field
zc2 = F.relu((s * (2 * (x_edit - fmn) / frg - 1)) @ W_enc + b_enc)[:, CONCEPT]
fire = zc > 0
print(f"concept {CONCEPT} mean act on firing nodes: before={zc[fire].mean():.4f} after={zc2[fire].mean():.4f} (expect ~0)")
