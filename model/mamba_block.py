"""
MambaBlock — Selective State Space Model with Parallel Associative Scan.

Key design choices:
  - Input-dependent B, C, dt (selective SSM, not fixed S4)
  - Zero-order hold (ZOH) discretisation
  - Blelloch parallel scan: O(N log N), fully GPU-parallelisable, no CUDA extensions
  - Causal Conv1d (k=4) for local context mixing before SSM
  - SiLU gating on output (Mamba paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ParallelScan(nn.Module):
    """
    Blelloch parallel prefix scan for linear recurrences:
        h_t = A_t * h_{t-1} + B_t * x_t

    Composition rule for two consecutive maps:
        (A2, B2) o (A1, B1) = (A2*A1, A2*B1 + B2)

    Args:
        A: (B, N, d_inner, d_state) — diagonal transition matrices
        B: (B, N, d_inner, d_state) — input matrices (already scaled by x)

    Returns:
        h: (B, N, d_inner, d_state) — all prefix hidden states
    """

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        B_sz, N, d_inner, d_state = A.shape
        device = A.device

        # Pad to next power of 2
        pad_len = (1 << math.ceil(math.log2(max(N, 2)))) - N
        if pad_len > 0:
            A = F.pad(A, (0, 0, 0, 0, 0, pad_len), value=1.0)  # pad A with 1 (identity)
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len), value=0.0)  # pad B with 0 (zero input)
        N_pad = A.shape[1]

        pa, pb = A.clone(), B.clone()

        # Up-sweep: reduce pairs up the tree
        stride = 1
        while stride < N_pad:
            idx = torch.arange(stride - 1, N_pad, step=2 * stride, device=device)
            left  = idx
            right = (idx + stride).clamp(max=N_pad - 1)

            # Compose: right absorbs left
            new_b = pa[:, right] * pb[:, left] + pb[:, right]  # A2*B1 + B2
            new_a = pa[:, right] * pa[:, left]                  # A2*A1

            pb[:, right] = new_b
            pa[:, right] = new_a
            stride *= 2

        # Down-sweep: distribute prefix back down
        stride = N_pad // 2
        while stride >= 1:
            idx = torch.arange(stride - 1, N_pad - stride, step=2 * stride, device=device)
            left  = idx
            right = idx + stride

            tmp_a = pa[:, left].clone()
            tmp_b = pb[:, left].clone()

            pa[:, left]  = pa[:, right]
            pb[:, left]  = pb[:, right]
            pa[:, right] = pa[:, right] * tmp_a
            pb[:, right] = pa[:, right] * tmp_b + pb[:, right]  # NOTE: pa already updated above
            stride //= 2

        # Trim back to original length
        return pb[:, :B_sz.bit_length() and N, :, :]  if False else pb[:, :N, :, :]


class MambaBlock(nn.Module):
    """
    One selective SSM block.

    Hyperparameters (following Mamba paper defaults):
        d_model  : input/output dimension
        d_inner  : expansion factor × d_model  (default 2x)
        d_state  : SSM state dimension          (default 16)
        d_conv   : causal conv kernel size      (default 4)
    """

    def __init__(self, d_model: int, d_inner: int = None, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_inner or 2 * d_model
        self.d_state  = d_state
        self.d_conv   = d_conv

        di = self.d_inner
        ds = self.d_state

        # Input projection: projects to content stream x and gate z
        self.in_proj  = nn.Linear(d_model, 2 * di, bias=False)

        # Causal depthwise Conv1d for local context (pads left only)
        self.conv1d   = nn.Conv1d(di, di, kernel_size=d_conv, groups=di, bias=True,
                                  padding=d_conv - 1)  # trim right in forward

        # Selective projections (produce B, C, log_dt from x)
        self.x_proj   = nn.Linear(di, ds + ds + 1, bias=False)  # B + C + dt_rank
        self.dt_proj  = nn.Linear(1, di, bias=True)             # expand dt scalar to di channels

        # A: fixed log-decay (learnable but time-invariant)
        A = torch.arange(1, ds + 1, dtype=torch.float32).unsqueeze(0).expand(di, -1)
        self.A_log    = nn.Parameter(torch.log(A))               # (di, ds)
        self.D        = nn.Parameter(torch.ones(di))             # skip connection scalar

        self.out_proj = nn.Linear(di, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.scan     = ParallelScan()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, N, d_model)
        Returns:
            (B, N, d_model)
        """
        B_sz, N, _ = u.shape
        residual = u

        # Split into content stream and gate
        xz   = self.in_proj(u)                          # (B, N, 2*di)
        x, z = xz.chunk(2, dim=-1)                      # (B, N, di) each

        # Causal Conv1d (transpose for Conv1d, trim causal padding)
        x_conv = self.conv1d(x.transpose(1, 2))         # (B, di, N + d_conv - 1)
        x      = x_conv[:, :, :N].transpose(1, 2)       # (B, N, di)  — causal trim
        x      = F.silu(x)

        # Selective projections
        proj   = self.x_proj(x)                         # (B, N, ds + ds + 1)
        B_ssm  = proj[..., :self.d_state]               # (B, N, ds)
        C_ssm  = proj[..., self.d_state:2*self.d_state] # (B, N, ds)
        log_dt = proj[..., -1:]                         # (B, N, 1)

        # Time step (always positive)
        dt     = F.softplus(self.dt_proj(log_dt))       # (B, N, di)

        # Discretise A using ZOH:  Ã_t = exp(Δ_t ⊗ A)
        A      = -torch.exp(self.A_log)                 # (di, ds) — negative for stability
        # Expand for broadcasting: (B, N, di, ds)
        A_bar  = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, N, di, ds)
        B_bar  = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)                      # (B, N, di, ds)
        # Scale by input x
        Bx     = B_bar * x.unsqueeze(-1)                # (B, N, di, ds)

        # Parallel scan → all hidden states
        h      = self.scan(A_bar, Bx)                   # (B, N, di, ds)

        # Read out via C
        y      = (h * C_ssm.unsqueeze(2)).sum(-1)       # (B, N, di)
        y      = y + self.D.unsqueeze(0).unsqueeze(0) * x  # skip connection D

        # SiLU gate and output projection
        y      = y * F.silu(z)                          # (B, N, di)
        out    = self.out_proj(y)                        # (B, N, d_model)

        return self.norm(out + residual)
