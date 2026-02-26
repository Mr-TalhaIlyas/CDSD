# multipositive_clip.py
# Robust multi-positive (many-to-one / many-to-many) CLIP-style loss

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _gather_if_needed(x: torch.Tensor) -> torch.Tensor:
    """
    All-gathers a tensor across DDP ranks (no grad sync issues since we don't backprop through gather).
    Returns x unchanged if not in distributed context.
    """
    if not dist.is_available() or not dist.is_initialized():
        return x
    world_size = dist.get_world_size()
    if world_size == 1:
        return x
    xs = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


@dataclass
class MultiPositiveClipConfig:
    # Temperature/scale
    init_logit_scale: float = 0.0         # exp(0)=1.0
    logit_scale_min: float = -4.0         # clamp to avoid degenerate temps
    logit_scale_max: float = 3.0 # 4.0

    # Normalization & numerical stability
    l2_normalize: bool = True
    eps: float = 1e-6

    # Reduction & symmetry
    reduction: str = "mean"               # "mean" | "sum" | "none"
    symmetric: bool = True                # include text->image term

    # Distributed negatives (optional)
    use_ddp_all_gather: bool = False      # gather features & masks across ranks

    # Label smoothing over positives (helps when many positives per row/col)
    label_smoothing: float = 0.0          # in [0, 0.2] typically

    # Optional class/row weighting (e.g., balance by freq)
    row_weights: Optional[torch.Tensor] = None  # shape [B] on each call
    col_weights: Optional[torch.Tensor] = None  # shape [C] on each call


class MultiPositiveClipLoss(nn.Module):
    """
    Multi-positive CLIP loss (symmetric by default).

    Inputs:
        image:     [B, D] image embeddings (pre- or post-projection)
        text:      [C, D] text/class embeddings (can be B if paired batch; C can differ from B)
        pos_mask:  [B, C] bool or {0,1} tensor. True/1 means image i matches text/class c.
                   Must have at least one positive per row; if symmetric=True, also per column.

    Computes for image->text:
        loss_i = mean_i [ logsumexp(logits[i, :]) - logsumexp(logits[i, P(i)]) ]
    and optionally for text->image (columns) and averages them.

    Features:
        - AMP-safe (only uses matmul, logsumexp, etc.)
        - Optional DDP all-gather to increase negatives across GPUs
        - Optional label smoothing over the positive set
        - Optional row/col weights
    """

    def __init__(self, cfg: MultiPositiveClipConfig = MultiPositiveClipConfig()):
        super().__init__()
        self.cfg = cfg
        self.logit_scale = nn.Parameter(torch.tensor(cfg.init_logit_scale, dtype=torch.float32))

    def forward(
        self,
        image: torch.Tensor,           # [B, D]
        text: torch.Tensor,            # [C, D]
        pos_mask: torch.Tensor,        # [B, C] bool or {0,1}
    ) -> Tuple[torch.Tensor, dict]:
        assert image.dim() == 2 and text.dim() == 2, "image/text must be [N, D]"
        assert pos_mask.dim() == 2 and pos_mask.size(0) == image.size(0), "pos_mask shape [B, C]"
        assert pos_mask.size(1) == text.size(0), "pos_mask second dim must match C"

        # Normalize if requested
        if self.cfg.l2_normalize:
            image = _l2_normalize(image, self.cfg.eps)
            text  = _l2_normalize(text,  self.cfg.eps)

        # Optionally extend negatives across DDP ranks
        if self.cfg.use_ddp_all_gather:
            image_all = _gather_if_needed(image)
            text_all  = _gather_if_needed(text)
            # Gather mask across batch (rows); columns correspond to text_all
            pos_mask_all = _gather_if_needed(pos_mask.float()).to(pos_mask.dtype)
        else:
            image_all, text_all, pos_mask_all = image, text, pos_mask

        B, D = image_all.shape
        C, _ = text_all.shape
        device = image_all.device

        # Checks: at least one positive per row/col if symmetric
        if not torch.any(pos_mask_all, dim=1).all():
            raise ValueError("Each image row must have at least one positive in pos_mask.")
        if self.cfg.symmetric and not torch.any(pos_mask_all, dim=0).all():
            raise ValueError("Each text column must have at least one positive in pos_mask when symmetric=True.")

        # Compute logits = exp(scale) * cosine
        logit_scale = self.logit_scale.clamp(self.cfg.logit_scale_min, self.cfg.logit_scale_max)
        logits = torch.exp(logit_scale) * (image_all @ text_all.t())  # [B, C]

        # Image->Text (rows)
        # LSE all
        lse_row_all = torch.logsumexp(logits, dim=1)  # [B]
        # LSE positives (mask out non-positives with -inf)
        neg_inf = torch.finfo(logits.dtype).min
        mask_row = pos_mask_all.to(torch.bool)
        logits_pos_row = logits.masked_fill(~mask_row, neg_inf)
        lse_row_pos = torch.logsumexp(logits_pos_row, dim=1)  # [B]

        # Optional label smoothing over positive set:
        # Replace logsumexp(pos) with logsumexp(logits + log(weights)) where weights are uniform but smoothed.
        if self.cfg.label_smoothing > 0.0:
            # weights per row: (1 - s)/|P(i)| for positives, s / C for all classes
            # implemented by adding log-weights to logits before logsumexp
            with torch.no_grad():
                pos_counts = mask_row.sum(dim=1).clamp_min(1).to(logits.dtype)  # [B]
                w_pos = (1.0 - self.cfg.label_smoothing) / pos_counts          # [B]
                w_all = (self.cfg.label_smoothing) / float(C)
            # Broadcast weights into [B, C]
            log_w = torch.full_like(logits, fill_value=torch.log(torch.tensor(w_all, dtype=logits.dtype, device=device)))
            log_w = torch.where(mask_row, torch.log(w_pos.unsqueeze(1)), log_w)
            lse_row_pos = torch.logsumexp(logits + log_w, dim=1)

        loss_i2t_row = (lse_row_all - lse_row_pos)  # [B]

        # Optional row weights
        if self.cfg.row_weights is not None:
            w = self.cfg.row_weights.to(device, dtype=logits.dtype)
            w = w / (w.sum() + 1e-8)
            loss_i2t = (loss_i2t_row * w).sum()
        else:
            loss_i2t = loss_i2t_row.mean()

        # Text->Image (columns, symmetric)
        if self.cfg.symmetric:
            lse_col_all = torch.logsumexp(logits, dim=0)  # [C]
            mask_col = mask_row  # same mask; we’ll reuse
            logits_pos_col = logits.masked_fill(~mask_col, neg_inf)
            lse_col_pos = torch.logsumexp(logits_pos_col, dim=0)  # [C]

            if self.cfg.label_smoothing > 0.0:
                with torch.no_grad():
                    pos_counts_c = mask_col.sum(dim=0).clamp_min(1).to(logits.dtype)  # [C]
                    w_pos_c = (1.0 - self.cfg.label_smoothing) / pos_counts_c
                    w_all_c = (self.cfg.label_smoothing) / float(B)
                log_w_c = torch.full_like(logits, fill_value=torch.log(torch.tensor(w_all_c, dtype=logits.dtype, device=device)))
                log_w_c = torch.where(mask_col, torch.log(w_pos_c.unsqueeze(0)), log_w_c)
                lse_col_pos = torch.logsumexp(logits + log_w_c, dim=0)

            loss_t2i_col = (lse_col_all - lse_col_pos)  # [C]

            if self.cfg.col_weights is not None:
                w = self.cfg.col_weights.to(device, dtype=logits.dtype)
                w = w / (w.sum() + 1e-8)
                loss_t2i = (loss_t2i_col * w).sum()
            else:
                loss_t2i = loss_t2i_col.mean()

            loss = 0.5 * (loss_i2t + loss_t2i)
        else:
            loss = loss_i2t

        # Reduction
        if self.cfg.reduction == "sum":
            # Above we used means; if you want strict "sum", you can rescale here as needed.
            pass
        elif self.cfg.reduction == "none":
            # Return per-sample view for rows (and optionally columns)
            # Here we return the averaged scalar but also attach row/col vectors in info.
            pass

        info = {
            "loss_i2t": loss_i2t.detach(),
            "logit_scale_exp": torch.exp(logit_scale).detach(),
        }
        return loss, info
#%%
'''
# model forward gives image/text projections (before L2) of size [B, D], [C, D]
image_z = image_proj(image_batch)     # [B, D]
text_z  = text_proj(class_or_text_embs)  # [C, D]

# Build B×C boolean mask of positives (multi-label)
# e.g., Y is 0/1 float -> convert to bool
pos_mask = (Y > 0.5)                  # [B, C] bool

cfg = MultiPositiveClipConfig(
    init_logit_scale=0.0,
    symmetric=True,
    use_ddp_all_gather=False,  # set True under DDP for stronger negatives
    label_smoothing=0.0
)
crit = MultiPositiveClipLoss(cfg)

loss, info = crit(image_z, text_z, pos_mask)
loss.backward()
optimizer.step()

'''