"""
FiLM — Feature-wise Linear Modulation.

Conditions the packet-sequence representation on a global network-context
vector (RTT, jitter, path state) after the first Mamba block.

    x_out[b, t, :] = (1 + γ(c_b)) ⊙ x[b, t, :] + β(c_b)

where γ, β ∈ R^{d_model} are produced by a 2-layer MLP from the context c.
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Args:
        d_model  : dimension of the sequence tensor to be modulated
        d_ctx    : dimension of the context vector
        hidden   : hidden size of the MLP that produces γ and β
    """

    def __init__(self, d_model: int, d_ctx: int, hidden: int = 64):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_ctx, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_model),   # → [γ, β]
        )
        # Zero-init so the module is an identity at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, N, d_model)   — packet sequence after first Mamba block
            ctx : (B, d_ctx)        — per-flow network context vector
        Returns:
            out : (B, N, d_model)   — modulated sequence
        """
        params = self.mlp(ctx)                        # (B, 2*d_model)
        gamma, beta = params.chunk(2, dim=-1)         # each (B, d_model)
        gamma = gamma.unsqueeze(1)                    # (B, 1, d_model)
        beta  = beta.unsqueeze(1)
        return (1.0 + gamma) * x + beta
