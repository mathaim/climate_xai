"""
Top-K Sparse Autoencoder with AuxK dead-latent recovery.

Architecture follows the OpenAI sparse autoencoder recipe:
  - Top-K activation (not L1 penalty)
  - Unit-norm decoder columns
  - Orthogonal gradient projection on decoder
  - AuxK loss to revive dead latents

Args:
    d_in: Input activation dimension (e.g. 512 for GraphCast layers)
    n_latents: Dictionary size (e.g. 4096)
    k_active: Number of top-k features kept per sample
    k_aux: Number of dead latents used in AuxK loss
    dead_window: Number of samples before a latent is considered dead
"""

import torch
import torch.nn as nn


def topk(x, k: int):
    """Keep top-k values per row, zero out the rest."""
    if k >= x.shape[1]:
        return x
    vals, idx = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x)
    mask.scatter_(1, idx, 1.0)
    return x * mask


class PlainSAE(nn.Module):
    """Top-K Sparse Autoencoder with AuxK loss."""

    def __init__(self, d_in=512, n_latents=4096, k_active=32, k_aux=512,
                 unit_norm_decoder=True, dead_window=3_000_000):
        super().__init__()
        self.d_in = d_in
        self.n_latents = n_latents
        self.k_active = k_active
        self.k_aux = k_aux
        self.unit_norm_decoder = unit_norm_decoder
        self.dead_window = dead_window
        self.eps = 1e-8

        self.enc = nn.Linear(d_in, n_latents, bias=False)
        self.dec = nn.Linear(n_latents, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))

        self.register_buffer("miss_counts", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer("dead_mask", torch.zeros(n_latents, dtype=torch.bool))

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.normal_(self.dec.weight, mean=0.0, std=1.0)
            W = self.dec.weight
            W.div_(W.norm(dim=0, keepdim=True).clamp_min(self.eps))
            self.enc.weight.copy_(W.t())
            self.b_pre.zero_()

    def forward(self, x):
        # Normalize inputs
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

        # Encode
        x_bar = x - self.b_pre
        code_pre = torch.relu(self.enc(x_bar))
        code = topk(code_pre, self.k_active)

        # Decode
        if self.unit_norm_decoder:
            W = self.dec.weight
            norms = W.norm(dim=0, keepdim=True).clamp_min(self.eps)
            recon = torch.addmm(self.b_pre, code, (W / norms).t())
        else:
            recon = self.dec(code) + self.b_pre

        # AuxK loss on dead latents
        if self.dead_mask.any():
            dead_code = code_pre * self.dead_mask.unsqueeze(0)
            aux_code = topk(dead_code, min(self.k_aux, dead_code.shape[1]))
            aux_recon = torch.addmm(
                torch.zeros_like(self.b_pre), aux_code, self.dec.weight.t()
            )
        else:
            aux_recon = torch.zeros_like(x)

        return recon, code, aux_recon

    def encode_topk(self, x):
        """Return top-k indices and values for sparse encoding."""
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
        x_bar = x - self.b_pre
        pre_acts = self.enc(x_bar)
        topk_values, topk_indices = torch.topk(pre_acts, self.k_active, dim=-1)
        return topk_indices, topk_values

    @torch.no_grad()
    def update_dead_mask(self, code, batch_size: int):
        active = (code > 0).any(dim=0).cpu()
        self.miss_counts[active] = 0
        self.miss_counts[~active] += batch_size
        self.dead_mask = self.miss_counts >= self.dead_window


# ── Optimizer helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def project_decoder_grads_orthogonal(model):
    """Project decoder gradients to be orthogonal to decoder columns."""
    W = model.dec.weight
    G = model.dec.weight.grad
    if G is None:
        return
    dots = (G * W).sum(dim=0, keepdim=True)
    norms2 = (W * W).sum(dim=0, keepdim=True).clamp_min(1e-8)
    G.sub_((dots / norms2) * W)


@torch.no_grad()
def renorm_decoder_columns(model):
    """Re-normalize decoder columns to unit norm."""
    W = model.dec.weight.data
    W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))
