"""
FlowContextEncoder — Context-Aware Flow Embeddings for Network Traffic Classification
=====================================================================================
Architecture:
  Packets (B, N, d_in)
    → pkt_embed  (Linear)
    → pos_enc    (sinusoidal)
    → MambaBlock × n_layers  (SSM with parallel associative scan)
      ↑ FiLM context fusion after block 0  (RTT/jitter/path-state)
    → + ctx_proj  (additive residual from context)
    → Masked GAP  (handles variable-length flows)
    → MLP Head
    → L2-norm  →  Embedding (B, 128)

Contrastive training: SupConLoss (supervised, tau=0.07)
Compatibility: pure PyTorch — no custom CUDA; runs on Windows/Linux/macOS
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1.  Parallel Associative Scan (Blelloch)
# ──────────────────────────────────────────────
def parallel_scan(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute prefix SSM states h_t = A_t * h_{t-1} + B_t for all t simultaneously.

    Args:
        A : (B, N, D)  — per-step decay  (\bar A, already discretised)
        B : (B, N, D)  — per-step input  (\bar B * x_t)
    Returns:
        h : (B, N, D)  — hidden states  h_1 … h_N

    Each time-step is a linear map  x -> A*x + B.
    Composition:  (A2,B2) ∘ (A1,B1)  =  (A2*A1,  A2*B1 + B2)
    A Blelloch tree computes all prefix compositions in O(N log N).
    """
    B_dim, N, D = A.shape
    # Pad to next power of two
    L = 1
    while L < N:
        L <<= 1
    if L > N:
        pad = L - N
        A = torch.cat([A, torch.ones(B_dim, pad, D, device=A.device, dtype=A.dtype)], dim=1)
        B = torch.cat([B, torch.zeros(B_dim, pad, D, device=B.device, dtype=B.dtype)], dim=1)

    pa, pb = A.clone(), B.clone()

    # ── Up-sweep (reduce) ──────────────────────
    step = 1
    while step < L:
        idx_r = torch.arange(step - 1, L, step * 2, device=A.device)
        idx_l = idx_r - step if step > 0 else idx_r  # left sibling
        # compose: right ∘ left
        new_pa = pa[:, idx_r] * pa[:, idx_l] if step > 0 else pa[:, idx_r]
        new_pb = pa[:, idx_r] * pb[:, idx_l] + pb[:, idx_r]
        pa[:, idx_r] = new_pa
        pb[:, idx_r] = new_pb
        step <<= 1

    # Zero out the identity at the root
    pa[:, -1] = torch.ones(B_dim, D, device=A.device, dtype=A.dtype)
    pb[:, -1] = torch.zeros(B_dim, D, device=B.device, dtype=B.dtype)

    # ── Down-sweep (scan) ─────────────────────
    step = L >> 1
    while step >= 1:
        idx_r = torch.arange(step - 1, L, step * 2, device=A.device)
        idx_l = idx_r - step
        tmp_a = pa[:, idx_l].clone()
        tmp_b = pb[:, idx_l].clone()
        pa[:, idx_l] = pa[:, idx_r]
        pb[:, idx_l] = pb[:, idx_r]
        pa[:, idx_r] = pa[:, idx_r] * tmp_a
        pb[:, idx_r] = pa[:, idx_r] * tmp_b + pb[:, idx_r]
        step >>= 1

    # pb now holds the exclusive prefix → shift by one gives inclusive
    h = torch.roll(pb, -1, dims=1)
    h[:, -1] = B[:, -1]  # last element: just B (no prior state)
    return h[:, :N]


# ──────────────────────────────────────────────
# 2.  Sinusoidal Positional Encoding
# ──────────────────────────────────────────────
class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, D)
        return x + self.pe[:, : x.size(1)]


# ──────────────────────────────────────────────
# 3.  MambaBlock  (selective SSM)
# ──────────────────────────────────────────────
class MambaBlock(nn.Module):
    """
    One Mamba block.

    Input:  x  (B, N, d_model)
    Output: x  (B, N, d_model)  — residual added internally

    Key operations
    --------------
    in_proj  : d_model → 2*d_inner  (content stream + gate)
    conv1d   : causal local mixing (k=4)
    x_proj   : produces input-dependent B_ssm, C_ssm, log_dt  (selectivity)
    dt_proj  : expands scalar Δ_t → d_inner channels
    A_log    : fixed log-decay; discretised as Ā = exp(Δ * A)
    scan     : parallel_scan gives all N hidden states
    y        : read-out via C_ssm; gated by silu(z)
    """

    def __init__(self, d_model: int, d_inner: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_inner
        self.d_state  = d_state

        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )
        # selective projections: Δ (scalar), B (d_state), C (d_state)
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)

        # A: fixed log-decay initialised to log(range)
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)  # (d_inner, d_state)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)                                    # (B, N, d_model)
        B_sz, N, _ = x.shape

        # ── split content and gate ─────────────
        xz   = self.in_proj(x)                             # (B, N, 2*d_inner)
        xc, z = xz.chunk(2, dim=-1)                        # each (B, N, d_inner)

        # ── causal conv1d ──────────────────────
        xc = self.conv1d(xc.transpose(1, 2))[:, :, :N].transpose(1, 2)  # (B, N, d_inner)
        xc = F.silu(xc)

        # ── selective parameters ───────────────
        proj   = self.x_proj(xc)                           # (B, N, 2*d_state+1)
        B_ssm  = proj[..., :self.d_state]                  # (B, N, d_state)
        C_ssm  = proj[..., self.d_state:2*self.d_state]    # (B, N, d_state)
        log_dt = proj[..., -1:]                            # (B, N, 1)  — scalar Δ

        dt     = F.softplus(self.dt_proj(log_dt))          # (B, N, d_inner)

        # ── discretise A ──────────────────────
        # Ā  = exp(Δ * A)   shape: (B, N, d_inner, d_state)
        A     = -torch.exp(self.A_log)                     # (d_inner, d_state)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)            # (B, N, d_inner, d_state)

        # ──  B̄ * x  ────────────────────────────
        # B_ssm: (B,N,d_state)  xc: (B,N,d_inner)
        # ΔB*x : (B, N, d_inner, d_state)
        dBx   = dt.unsqueeze(-1) * (xc.unsqueeze(-1) * B_ssm.unsqueeze(-2))

        # ── parallel scan ─────────────────────
        # Flatten d_inner * d_state into a single channel dim for the scan
        A_flat  = A_bar.view(B_sz, N, self.d_inner * self.d_state)
        dBx_flat= dBx.view(B_sz, N, self.d_inner * self.d_state)
        h_flat  = parallel_scan(A_flat, dBx_flat)          # (B, N, d_inner*d_state)
        h       = h_flat.view(B_sz, N, self.d_inner, self.d_state)

        # ── selective read-out via C ───────────
        # y_t = Σ_s C_ssm_t[s] * h_t[i,s]  for each channel i
        # h: (B,N,d_inner,d_state)  C_ssm: (B,N,d_state)
        y = (h * C_ssm.unsqueeze(-2)).sum(-1)              # (B, N, d_inner)
        y = y + self.D * xc                                # skip connection

        # ── gate and project ──────────────────
        y = y * F.silu(z)
        y = self.out_proj(y)                               # (B, N, d_model)
        return y + residual


# ──────────────────────────────────────────────
# 4.  FiLM Context Fusion
# ──────────────────────────────────────────────
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Applies  x <- (1 + gamma(c)) * x + beta(c)
    where gamma, beta are MLPs of the context vector c.
    """

    def __init__(self, d_model: int, d_ctx: int):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(d_ctx, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.beta  = nn.Sequential(
            nn.Linear(d_ctx, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        nn.init.zeros_(self.gamma[-1].weight)
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D)  ctx: (B,D_ctx)
        g = self.gamma(ctx).unsqueeze(1)   # (B,1,D)
        b = self.beta(ctx).unsqueeze(1)    # (B,1,D)
        return (1 + g) * x + b


# ──────────────────────────────────────────────
# 5.  FlowContextEncoder
# ──────────────────────────────────────────────
class FlowContextEncoder(nn.Module):
    """
    Full encoder: packets + network context → L2-normalised embedding.

    Args:
        d_in     : per-packet feature dimension  (e.g. 9)
        d_ctx    : context vector dimension      (e.g. 8: RTT, jitter, ...)
        d_model  : internal model width          (default 128)
        d_inner  : SSM inner width               (default 256)
        d_state  : SSM state size                (default 16)
        n_layers : number of MambaBlocks         (default 4)
        d_emb    : output embedding dimension    (default 128)
        max_len  : max sequence length           (default 512)
    """

    def __init__(
        self,
        d_in: int = 9,
        d_ctx: int = 8,
        d_model: int = 128,
        d_inner: int = 256,
        d_state: int = 16,
        n_layers: int = 4,
        d_emb: int = 128,
        max_len: int = 512,
    ):
        super().__init__()

        self.pkt_embed = nn.Linear(d_in, d_model)
        self.pos_enc   = SinusoidalPosEnc(d_model, max_len)

        self.layers    = nn.ModuleList([
            MambaBlock(d_model, d_inner, d_state) for _ in range(n_layers)
        ])
        # FiLM fusion injected after the first block
        self.film      = FiLM(d_model, d_ctx)
        # Additive context residual (global)
        self.ctx_proj  = nn.Sequential(nn.Linear(d_ctx, d_model), nn.SiLU())

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_emb),
        )

    def forward(
        self,
        pkts: torch.Tensor,           # (B, N, d_in)
        ctx:  torch.Tensor,           # (B, d_ctx)
        mask: torch.Tensor | None = None,  # (B, N)  True = valid packet
    ) -> torch.Tensor:
        """
        Returns
        -------
        emb : (B, d_emb)  —  L2-normalised embedding
        """
        x = self.pkt_embed(pkts)       # (B, N, d_model)
        x = self.pos_enc(x)

        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x = self.film(x, ctx)  # inject network context after first block

        # Global additive context residual
        x = x + self.ctx_proj(ctx).unsqueeze(1)

        # Masked Global Average Pooling
        pooled = (x * mask.float().unsqueeze(-1)).sum(dim=1) \
               / mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, d_model)

        emb = self.head(pooled)        # (B, d_emb)
        return F.normalize(emb, dim=-1)


# ──────────────────────────────────────────────
# 6.  Supervised Contrastive Loss
# ──────────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss  (Khosla et al., NeurIPS 2020).

    For each anchor i, all flows with the same label are positives.
    Temperature tau=0.07 sharpens the similarity distribution.

    L = - 1/|P(i)| * sum_{p in P(i)}  log  exp(z_i . z_p / tau)
                                            ──────────────────────
                                            sum_{k≠i} exp(z_i . z_k / tau)
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # z: (B, D) L2-normalised;  labels: (B,)
        sim   = z @ z.T / self.tau                         # (B, B)
        # Remove diagonal (self-similarity)
        mask_self = ~torch.eye(sim.size(0), dtype=torch.bool, device=z.device)
        # Positive mask: same label, not self
        mask_pos  = (labels.unsqueeze(1) == labels.unsqueeze(0)) & mask_self  # (B,B)

        # Numerical stability: subtract row max
        sim = sim - sim.detach().max(dim=1, keepdim=True).values

        exp_sim   = torch.exp(sim) * mask_self.float()     # zero out diagonal
        log_prob  = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))

        # Average over positives per anchor
        n_pos    = mask_pos.float().sum(dim=1)             # (B,)
        has_pos  = n_pos > 0
        if not has_pos.any():
            return torch.tensor(0.0, requires_grad=True, device=z.device)

        loss = -(log_prob * mask_pos.float()).sum(dim=1)   # (B,)
        loss = (loss[has_pos] / n_pos[has_pos]).mean()
        return loss


# ──────────────────────────────────────────────
# 7.  Smoke test / minimal training loop
# ──────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyper-parameters (small for quick test)
    B, N, d_in, d_ctx = 32, 64, 9, 8
    d_model, d_inner, d_state, n_layers, d_emb = 64, 128, 8, 2, 128

    model  = FlowContextEncoder(
        d_in, d_ctx, d_model, d_inner, d_state, n_layers, d_emb
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: ~{n_params:,}")

    criterion = SupConLoss(tau=0.07)
    opt       = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)

    # Synthetic dataset: 5 classes, variable-length flows
    N_CLASSES = 5

    def make_batch():
        pkts   = torch.randn(B, N, d_in).to(device)
        ctx    = torch.randn(B, d_ctx).to(device)
        labels = torch.randint(0, N_CLASSES, (B,)).to(device)
        # Random masks (each flow has 16..N valid packets)
        lengths = torch.randint(16, N + 1, (B,))
        mask = torch.arange(N).unsqueeze(0) < lengths.unsqueeze(1)  # (B,N)
        return pkts, ctx, mask.to(device), labels

    # Training loop
    model.train()
    for epoch in range(1, 16):
        pkts, ctx, mask, labels = make_batch()
        emb  = model(pkts, ctx, mask)
        loss = criterion(emb, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # Diagnostic: intra- vs inter-class cosine similarity
        with torch.no_grad():
            sim = emb @ emb.T
            pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
            neg_mask = ~pos_mask
            pos_mask.fill_diagonal_(False)
            neg_mask.fill_diagonal_(False)
            intra = sim[pos_mask].mean().item() if pos_mask.any() else 0.0
            inter = sim[neg_mask].mean().item() if neg_mask.any() else 0.0
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} "
              f"| Intra sim: {intra:.4f} | Inter sim: {inter:.4f}")

    # Inference check
    model.eval()
    with torch.no_grad():
        test_pkts = torch.randn(4, 32, d_in).to(device)
        test_ctx  = torch.randn(4, d_ctx).to(device)
        out = model(test_pkts, test_ctx)
    print(f"\nOutput shape : {out.shape}")
    print(f"L2 norms     : {out.norm(dim=-1).tolist()}")
