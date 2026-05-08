"""
FlowContextEncoder  — full pipeline.

  Packets  (B, N, d_in)  +  Context (B, d_ctx)  +  Mask (B, N)
      │
      ├── pkt_embed   : Linear(d_in  → d_model)
      ├── pos_enc     : learnable positional embedding (max_len, d_model)
      ├── MambaBlock  : block 0
      ├── FiLM        : context fusion after block 0
      ├── MambaBlock  : blocks 1 … n_layers-1
      ├── ctx_proj    : Linear(d_ctx → d_model) added to every position
      ├── norm        : LayerNorm(d_model)
      ├── masked_gap  : masked global average pool  → (B, d_model)
      ├── proj_head   : 2-layer MLP  → (B, d_embed)
      └── L2 norm     → embedding  (B, d_embed)  on unit hypersphere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import MambaBlock
from .film import FiLM


class FlowContextEncoder(nn.Module):
    """
    Args:
        d_in      : number of per-packet input features
        d_ctx     : dimension of the network-context vector  (RTT, jitter, …)
        d_model   : internal model dimension
        d_embed   : final embedding dimension (L2-normalised)
        n_layers  : number of MambaBlocks
        d_state   : SSM state size
        d_conv    : causal conv kernel size in MambaBlock
        expand    : inner expansion ratio in MambaBlock
        max_len   : max supported sequence length (for positional embedding)
        proj_mult : MLP head hidden dim = d_model * proj_mult
    """

    def __init__(
        self,
        d_in: int = 6,
        d_ctx: int = 4,
        d_model: int = 64,
        d_embed: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_len: int = 256,
        proj_mult: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_embed = d_embed
        self.n_layers = n_layers

        # --- Input projection ---
        self.pkt_embed = nn.Linear(d_in, d_model)

        # --- Learnable positional embeddings ---
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # --- Mamba layers ---
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        # --- FiLM context fusion (applied after block 0) ---
        self.film = FiLM(d_model, d_ctx)

        # --- Additive context injection (applied to all positions before GAP) ---
        self.ctx_proj = nn.Linear(d_ctx, d_model, bias=False)

        # --- Final normalisation ---
        self.norm = nn.LayerNorm(d_model)

        # --- Projection head: 2-layer MLP with GELU ---
        h_dim = d_model * proj_mult
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, d_embed),
        )

    def forward(
        self,
        packets: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            packets : (B, N, d_in)   — per-packet feature matrix
            ctx     : (B, d_ctx)     — per-flow context vector  (RTT, jitter, …)
            mask    : (B, N) bool    — True for real packets, False for padding
                      If None, all positions treated as real.
        Returns:
            z : (B, d_embed)  — L2-normalised flow embedding
        """
        B, N, _ = packets.shape
        device   = packets.device

        # --- Packet embedding + positional encoding ---
        x = self.pkt_embed(packets)                             # (B, N, d_model)
        pos = torch.arange(N, device=device).unsqueeze(0)      # (1, N)
        x = x + self.pos_emb(pos)                              # broadcast over B

        # --- Default mask: all valid ---
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # --- Mamba blocks with FiLM fusion after block 0 ---
        for i, block in enumerate(self.blocks):
            x = block(x)                     # (B, N, d_model)
            if i == 0:
                x = self.film(x, ctx)        # context modulation

        # --- Additive context injection before pooling ---
        ctx_vec = self.ctx_proj(ctx).unsqueeze(1)               # (B, 1, d_model)
        x = x + ctx_vec                                         # broadcast over N

        x = self.norm(x)                                        # (B, N, d_model)

        # --- Masked Global Average Pooling ---
        m = mask.float().unsqueeze(-1)                          # (B, N, 1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, d_model)

        # --- Projection head + L2 normalisation ---
        z = self.proj_head(pooled)                              # (B, d_embed)
        z = F.normalize(z, p=2, dim=-1)                        # unit hypersphere
        return z

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
