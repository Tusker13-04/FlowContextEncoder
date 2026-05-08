"""
Supervised Contrastive Loss.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

Key properties:
  - All flows with the same class label are positives for each other.
  - Temperature tau=0.07 sharpens the similarity distribution.
  - Anchors with no positive in the batch are skipped gracefully.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Args:
        temperature: tau  (default 0.07)
    Input:
        z:      (B, d_emb)  — L2-normalised embeddings
        labels: (B,)        — integer class labels
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        device = z.device

        # (B, B) cosine similarity matrix (embeddings are already L2-normed)
        sim = z @ z.T / self.tau

        # Numerical stability: subtract row max
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Mask out self-similarity on diagonal
        mask_self = ~torch.eye(B, dtype=torch.bool, device=device)

        # Positive mask: same label, different sample
        label_mat = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        pos_mask  = label_mat & mask_self

        # Skip anchors with no positive in the batch
        has_pos   = pos_mask.sum(1) > 0
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Log-sum-exp denominator (all non-self pairs)
        exp_sim   = torch.exp(sim) * mask_self.float()
        log_denom = torch.log(exp_sim.sum(1).clamp(min=1e-9))

        # Per-anchor loss: mean over positives
        per_pos   = (sim - log_denom.unsqueeze(1)) * pos_mask.float()
        n_pos     = pos_mask.float().sum(1).clamp(min=1)
        per_anchor = -(per_pos.sum(1) / n_pos)

        return per_anchor[has_pos].mean()
