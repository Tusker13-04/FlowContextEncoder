"""
MambaBlock  — one selective SSM layer.

Data flow (per block):
  x_in  (B, N, d_model)
    │
    ├─ in_proj  → [x_stream, z_gate]   (B, N, 2*d_inner)  — expand & gate split
    │
    ├─ conv1d   → x_stream             causal local mixing  (kernel=4, groups=d_inner)
    │
    ├─ x_proj   → B_ssm, C_ssm, Δ_raw  input-dependent SSM params per timestep
    │
    ├─ dt_proj  → Δ  (softplus)         expand scalar Δ to all channels
    │
    ├─ ZOH discretise:
    │     Ā = exp(Δ ⊙ A)              A = -exp(A_log)  fixed learnable
    │     B̄ = Δ ⊙ B_ssm ⊙ x_stream
    │
    ├─ parallel_scan(Ā, B̄)  → h       (B, N, d_inner)
    │
    ├─ y = Σ_d (h ⊙ C_ssm)            selective readout
    │
    ├─ y = y * silu(z_gate)            gating
    │
    └─ out_proj → x_out  (B, N, d_model)  + residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .ssm import parallel_scan


class MambaBlock(nn.Module):
    """
    Args:
        d_model   : model dimension (must equal d_inner when expand=1)
        d_inner   : inner / expanded dimension  (default: d_model * expand)
        d_state   : SSM state size (rank of A, B, C matrices)
        d_conv    : causal conv1d kernel size
        expand    : expansion ratio for d_inner
        dt_rank   : rank of Δ projection  ('auto' → ceil(d_model/16))
        dt_min/max: clamp range for softplus Δ initialisation
        dt_scale  : init scale for dt_proj bias
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.d_inner = d_model * expand

        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # --- Input projection: x + gate in one shot ---
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # --- Causal depthwise conv over the sequence ---
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,   # causal: trim right later
            bias=True,
        )

        # --- Selective parameter projections ---
        # B_ssm (d_state), C_ssm (d_state), dt_raw (dt_rank)  — all per timestep
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt_proj: dt_rank → d_inner  (with init to spread Δ in [dt_min, dt_max])
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Bias initialised so softplus(bias) ≈ uniform in [dt_min, dt_max]
        dt_bias = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # Inverse softplus
        self.dt_proj.bias = nn.Parameter(
            torch.log(torch.expm1(dt_bias)), requires_grad=True
        )

        # --- Fixed learnable log-decay A ---
        # A = -exp(A_log) ensures A < 0 (stable decay)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        A = A.repeat(self.d_inner, 1)                                         # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A), requires_grad=True)

        # --- D: skip / residual connection in SSM output ---
        self.D = nn.Parameter(torch.ones(self.d_inner), requires_grad=True)

        # --- Output projection ---
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # --- Layer norm before each block ---
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, d_model)
        Returns:
            out : (B, N, d_model)   residual added internally
        """
        residual = x
        x = self.norm(x)   # pre-norm

        B, N, _ = x.shape

        # 1. Expand + gate split
        xz = self.in_proj(x)                   # (B, N, 2*d_inner)
        x_s, z = xz.chunk(2, dim=-1)           # each (B, N, d_inner)

        # 2. Causal conv1d  (operates on channel dim, sequence is "time")
        x_s = x_s.transpose(1, 2)              # (B, d_inner, N)
        x_s = self.conv1d(x_s)[:, :, :N]       # trim right padding → causal
        x_s = F.silu(x_s)
        x_s = x_s.transpose(1, 2)              # (B, N, d_inner)

        # 3. Selective projections
        xp  = self.x_proj(x_s)                # (B, N, dt_rank + 2*d_state)
        dt_raw  = xp[..., :self.dt_rank]                          # (B, N, dt_rank)
        B_ssm   = xp[..., self.dt_rank : self.dt_rank + self.d_state]  # (B, N, d_state)
        C_ssm   = xp[..., self.dt_rank + self.d_state :]          # (B, N, d_state)

        # 4. Discretise Δ (softplus ensures positivity)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, N, d_inner)

        # 5. ZOH discretise A and B
        A = -torch.exp(self.A_log.float())     # (d_inner, d_state)  < 0
        # Ā_t = exp(Δ_t ⊙ A)  — shape broadcast: (B, N, d_inner, d_state)
        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )                                       # (B, N, d_inner, d_state)
        # B̄_t = Δ_t ⊙ x_t ⊙ B_t  — note B_ssm is (B, N, d_state)
        dB = (
            dt.unsqueeze(-1)
            * x_s.unsqueeze(-1)
            * B_ssm.unsqueeze(2)
        )                                       # (B, N, d_inner, d_state)

        # 6. Parallel scan across N for each (d_inner, d_state) independently
        # Flatten the last two dims so parallel_scan sees (B, N, d_inner*d_state)
        dA_flat = dA.reshape(B, N, self.d_inner * self.d_state)
        dB_flat = dB.reshape(B, N, self.d_inner * self.d_state)

        h_flat = parallel_scan(dA_flat, dB_flat)   # (B, N, d_inner*d_state)
        h = h_flat.reshape(B, N, self.d_inner, self.d_state)  # (B, N, d_inner, d_state)

        # 7. Selective readout:  y_t = Σ_s  h_t[s] * C_t[s]
        y = (h * C_ssm.unsqueeze(2)).sum(-1)    # (B, N, d_inner)

        # 8. D skip connection
        y = y + x_s * self.D.unsqueeze(0).unsqueeze(0)

        # 9. Gate
        y = y * F.silu(z)                       # (B, N, d_inner)

        # 10. Project back to d_model and add residual
        out = self.out_proj(y) + residual       # (B, N, d_model)
        return out
