import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


def get_wsd_scheduler(
    optimizer, n_steps, end_lr_factor=0.1, n_warmup_steps=None, percent_cooldown=0.1
):
    if n_warmup_steps is None:
        n_warmup_steps = 0.05 * n_steps

    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / n_warmup_steps
        elif step < (1 - percent_cooldown) * n_steps:
            return 1
        else:
            return 1 - (1 - end_lr_factor) * min(
                (step - (1 - percent_cooldown) * n_steps),
                (1 - percent_cooldown) * n_steps,
            ) / (percent_cooldown * n_steps + 1e-2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class RunningAvgNormalizer(nn.Module):
    def __init__(self, alpha=0.99):
        super().__init__()
        self.running_avg = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.alpha = alpha

    @torch.no_grad()
    def normalize(self, x, update=False):
        if update is True:
            with torch.no_grad():
                if self.running_avg is None:
                    self.running_avg.data = x.norm(dim=-1).mean()
                else:
                    self.running_avg.data = (
                        self.alpha * self.running_avg
                        + (1 - self.alpha) * x.norm(dim=-1).mean()
                    )
        return x * (np.sqrt(x.shape[-1]) / self.running_avg.detach())

    @torch.no_grad()
    def unnormalize(self, x):
        return x * (self.running_avg.detach() / np.sqrt(x.shape[-1]))


class MatryoshkaSAE(nn.Module):
    """
    Matryoshka SAE following Bussmann et al. (2025).

    Key differences from the Nabeshima implementation:
      1. BatchTopK sparsity (hard top-k per batch) instead of L1 + adaptive controller
      2. Fixed nested group sizes instead of random Pareto-sampled prefixes
      3. No sparsity loss term — sparsity is enforced entirely by BatchTopK

    Args:
        d_model: Input dimension (512 for GraphCast layer 8)
        n_latents: Total dictionary size (e.g. 4096)
        group_sizes: Fixed nested group boundaries (e.g. [256, 512, 1024, 2048, 4096])
                     Each must include all latents up to that index.
                     The last entry must equal n_latents.
        target_l0: Target average number of active features per input
        n_steps: Total training steps (for scheduler)
        lr: Learning rate (paper uses 3e-2)
    """

    def __init__(
        self,
        d_model,
        n_latents,
        group_sizes,
        target_l0,
        n_steps,
        lr=3e-2,
        permute_latents=True,
    ):
        super().__init__()

        self.W_enc = nn.Parameter(torch.randn(d_model, n_latents) / (np.sqrt(d_model)))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(
            (0.1 * self.W_enc.data / self.W_enc.data.norm(dim=0, keepdim=True)).T
        )
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.n_latents = n_latents
        self.d_model = d_model
        self.group_sizes = group_sizes
        self.n_groups = len(group_sizes)
        self.target_l0 = target_l0
        self.n_steps = n_steps

        assert group_sizes[-1] == n_latents, \
            f"Last group size must equal n_latents ({n_latents}), got {group_sizes[-1]}"
        for i in range(1, len(group_sizes)):
            assert group_sizes[i] > group_sizes[i-1], \
                f"Group sizes must be strictly increasing, got {group_sizes}"

        self.normalizer = RunningAvgNormalizer()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.9375))
        self.scaler = torch.amp.GradScaler("cuda")
        self.scheduler = get_wsd_scheduler(
            self.optimizer,
            n_steps=n_steps,
            n_warmup_steps=100,
            percent_cooldown=0.2,
            end_lr_factor=0.1,
        )

        self.permute_latents = permute_latents
        if self.permute_latents:
            self.sq_act_running_avg = nn.Parameter(
                torch.zeros(self.n_latents), requires_grad=False
            )

        self.step_count = 0

    @property
    def dtype(self):
        return self.W_enc.dtype

    @property
    def device(self):
        return self.W_enc.device

    def _batch_topk(self, pre_acts, k_per_sample):
        """
        BatchTopK: keep the top (batch_size * k_per_sample) activations
        across the entire batch, zero the rest.

        This enforces average sparsity of k_per_sample per sample,
        but individual samples can have more or fewer active features.
        """
        batch_size = pre_acts.shape[0]
        total_k = int(batch_size * k_per_sample)
        flat = pre_acts.reshape(-1)
        topk_values, topk_indices = torch.topk(flat, total_k)
        mask = torch.zeros_like(flat)
        mask.scatter_(0, topk_indices, 1.0)
        mask = mask.reshape(pre_acts.shape)
        return pre_acts * mask

    @torch.no_grad()
    def get_acts(self, x, indices=None, normalize=True):
        if normalize:
            x = self.normalizer.normalize(x, update=False)
        if isinstance(indices, int):
            indices = [indices]
        if indices is None:
            preacts = x @ self.W_enc + self.b_enc
            acts = self._batch_topk(F.relu(preacts), self.target_l0)
            acts = torch.einsum("...d,d->...d", acts, self.W_dec.norm(dim=1))
        else:
            preacts = x @ self.W_enc[:, indices] + self.b_enc[indices]
            acts = F.relu(preacts)
            acts = torch.einsum(
                "...d,d->...d", acts, self.W_dec[indices].norm(dim=1)
            )
        return self.normalizer.unnormalize(acts)

    def step(self, x, return_metrics=False):
        x = self.normalizer.normalize(x, update=True)

        # Compute raw pre-activations (no ReLU — BatchTopK handles sparsity)
        raw_pre_acts = x @ self.W_enc + self.b_enc

        # Apply BatchTopK: keeps top batch_size*target_l0 values, zeros the rest
        # This naturally selects positive values without needing ReLU
        acts_full = self._batch_topk(raw_pre_acts, self.target_l0)

        # Track L0
        with torch.no_grad():
            avg_l0 = (acts_full > 0).float().sum(dim=-1).mean().item()

        # Compute reconstruction for each nested group
        # Group boundaries: [0, group_sizes[0]], [0, group_sizes[1]], ...
        # Each group uses only the first group_sizes[i] latents
        group_recons = []
        for g_size in self.group_sizes:
            acts_group = acts_full[:, :g_size]
            recon = acts_group @ self.W_dec[:g_size] + self.b_dec
            group_recons.append(recon)

        # Compute MSE loss for each group and average
        group_losses = []
        for recon in group_recons:
            mse = ((recon - x) ** 2).sum(dim=-1).mean()
            group_losses.append(mse)

        loss = torch.stack(group_losses).mean()

        result = {"loss": loss, "avg_l0": avg_l0, "sparsity_scale": 0.0}

        if return_metrics:
            with torch.no_grad():
                tot_var = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean()
                for i, gl in enumerate(group_losses):
                    result[f"group_{self.group_sizes[i]}_fvu"] = (gl / tot_var).item()
                result["last_group_fvu"] = (group_losses[-1] / tot_var).item()

        # Update weights
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # Permute latents by importance within each group
        if self.permute_latents:
            with torch.no_grad():
                W_dec_norms = self.W_dec.norm(dim=1)
                normed_acts = acts_full * W_dec_norms[None]
                mse_contributions = (normed_acts ** 2).mean(dim=0)

                self.sq_act_running_avg.data = (
                    0.95 * self.sq_act_running_avg.data + 0.05 * mse_contributions
                )

                # Permute within each group to maintain group boundaries
                # but sort by importance within groups
                latent_perm = torch.arange(self.n_latents, device=self.device)
                prev_boundary = 0
                for g_size in self.group_sizes:
                    group_scores = self.sq_act_running_avg.data[prev_boundary:g_size]
                    group_perm = torch.argsort(group_scores, descending=True) + prev_boundary
                    latent_perm[prev_boundary:g_size] = group_perm
                    prev_boundary = g_size

                self.W_dec.data = self.W_dec.data[latent_perm]
                self.W_enc.data = self.W_enc.data[:, latent_perm]
                self.b_enc.data = self.b_enc.data[latent_perm]

                # Permute optimizer state
                for param_group in self.optimizer.param_groups:
                    for param in param_group["params"]:
                        state = self.optimizer.state.get(param, {})
                        if param is self.W_dec:
                            if "exp_avg" in state:
                                state["exp_avg"] = state["exp_avg"][latent_perm]
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"] = state["exp_avg_sq"][latent_perm]
                        elif param is self.W_enc:
                            if "exp_avg" in state:
                                state["exp_avg"] = state["exp_avg"][:, latent_perm]
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"] = state["exp_avg_sq"][:, latent_perm]
                        elif param is self.b_enc:
                            if "exp_avg" in state:
                                state["exp_avg"] = state["exp_avg"][latent_perm]
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"] = state["exp_avg_sq"][latent_perm]

        self.step_count += 1
        return result
