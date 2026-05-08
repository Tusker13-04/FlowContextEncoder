"""
FlowContextEncoder — full model.

Architecture:
    pkt_embed  →  pos_enc  →  MambaBlock[0]
    → FiLM(context)  →  MambaBlock[1..n-1]
    → ctx_residual  →  MaskedGAP  →  MLP head  →  L2-norm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_block import MambaBlock


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal PE — no learned parameters, generalises to any N."""
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation.
    Conditions entire sequence on network context c:
        x = (1 + gamma(c)) * x + beta(c)
    Placed after Block 0 so context modulates an already-rich packet representation.
    """
    def __init__(self, d_ctx: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_ctx, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),  # outputs [gamma, beta]
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)  # init as identity (no shift at start)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, N, d_model)
        ctx: (B, d_ctx)
        """
        gb     = self.mlp(ctx)                     # (B, 2*d_model)
        gamma, beta = gb.chunk(2, dim=-1)          # (B, d_model) each
        return (1 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


class FlowContextEncoder(nn.Module):
    """
    Full flow encoder.

    Args:
        d_in    : per-packet feature dimension          (default 8)
        d_ctx   : network context dimension             (default 4: RTT, jitter, loss, tput)
        d_model : internal model dimension              (default 128)
        d_emb   : output embedding dimension            (default 128)
        n_layers: number of Mamba blocks                (default 4)
        d_state : SSM state dimension per MambaBlock    (default 16)
        max_len : maximum flow length (packets)         (default 256)
        dropout : dropout probability                   (default 0.1)
    """

    def __init__(
        self,
        d_in:    int = 8,
        d_ctx:   int = 4,
        d_model: int = 128,
        d_emb:   int = 128,
        n_layers:int = 4,
        d_state: int = 16,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pkt_embed = nn.Linear(d_in, d_model)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Mamba blocks
        self.blocks    = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)
        ])

        # FiLM fusion after block 0
        self.film      = FiLMFusion(d_ctx, d_model)

        # Late context residual (additive reinforcement after all blocks)
        self.ctx_proj  = nn.Linear(d_ctx, d_model)

        # MLP projection head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_emb),
        )

    def forward(
        self,
        packets: torch.Tensor,
        context: torch.Tensor,
        mask:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            packets: (B, N, d_in)
            context: (B, d_ctx)
            mask:    (B, N)  — True for real packets, False for padding
        Returns:
            z: (B, d_emb)  — L2-normalised flow embedding
        """
        x = self.pos_enc(self.pkt_embed(packets))   # (B, N, d_model)

        x = self.blocks[0](x)                       # first Mamba block
        x = self.film(x, context)                   # inject RTT/jitter via FiLM

        for blk in self.blocks[1:]:                 # remaining blocks
            x = blk(x)

        # Late context residual — broadcast context over sequence
        x = x + self.ctx_proj(context).unsqueeze(1)

        # Masked Global Average Pooling
        m = mask.float().unsqueeze(-1)              # (B, N, 1)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)  # (B, d_model)

        # Project and L2-normalise
        z = self.head(pooled)                       # (B, d_emb)
        return F.normalize(z, p=2, dim=-1)
