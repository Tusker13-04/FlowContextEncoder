"""
Supervised Contrastive Loss  (Khosla et al., NeurIPS 2020)

For each anchor i, all flows with the same app-type label form the positive
set P(i).  The loss pushes anchors toward their positives and away from all
negatives on the unit hypersphere:

    L_i = - 1/|P(i)| * Σ_{p∈P(i)}  log [
               exp(z_i · z_p / τ)
              ─────────────────────────────────
              Σ_{k≠i} exp(z_i · z_k / τ)
          ]

    L = mean over all valid anchors (those with at least one positive).

Assumes z is already L2-normalised (cosine similarity = dot product).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Args:
        temperature : τ — sharpness of the distribution  (default 0.07)
        reduction   : 'mean' | 'sum' | 'none'
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction   = reduction

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z      : (B, d_embed)  — L2-normalised embeddings
            labels : (B,)          — integer class labels
        Returns:
            loss   : scalar (or per-sample tensor if reduction='none')
        """
        B = z.shape[0]
        device = z.device

        # Cosine similarity matrix  (B, B)
        sim = z @ z.T / self.temperature

        # Numerical stability: subtract row max before exp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Mask: same label = positive pair  (B, B)
        labels = labels.view(-1, 1)                         # (B, 1)
        pos_mask = (labels == labels.T).float().to(device)  # (B, B)  1 if same class

        # Remove self-comparisons from both positive mask and denominator
        eye = torch.eye(B, device=device)
        pos_mask = pos_mask * (1 - eye)
        neg_mask = 1 - eye                                  # all except self

        # Denominator: sum of exp over all non-self pairs
        exp_sim = torch.exp(sim) * neg_mask                 # zero out self
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)   # (B,)

        # Per-anchor loss
        # sum of log-sim over positives, normalised by |P(i)|
        pos_count = pos_mask.sum(dim=1)                     # (B,)
        has_pos   = pos_count > 0                           # skip anchors with no positive

        per_anchor = -(pos_mask * sim).sum(dim=1) / pos_count.clamp(min=1) + log_denom
        per_anchor = per_anchor[has_pos]                    # (M,) where M <= B

        if per_anchor.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if self.reduction == "mean":
            return per_anchor.mean()
        elif self.reduction == "sum":
            return per_anchor.sum()
        return per_anchor
