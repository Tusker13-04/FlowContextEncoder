"""
Parallel Associative Scan (Blelloch tree) for the S6 / Mamba SSM core.

Each timestep defines a linear map  x -> A_t * x + B_t.
The scan computes all prefix compositions in O(N log N) parallel steps,
returning the cumulative hidden state h_t for every position t.

Works on CPU and any CUDA device with pure PyTorch — no custom extensions.
"""

import torch
import torch.nn as nn
import math


def _compose(a2: torch.Tensor, b2: torch.Tensor,
             a1: torch.Tensor, b1: torch.Tensor):
    """
    Compose two linear maps:  (A2, B2) o (A1, B1)  =  (A2*A1, A2*B1 + B2)
    All tensors: (B, d)  — broadcasted element-wise (diagonal A).
    """
    return a2 * a1, a2 * b1 + b2


def parallel_scan(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix scan over sequence axis.

    Args:
        A : (B, N, d)  — discretised decay  exp(Δ * A_log)  in (0, 1)
        B : (B, N, d)  — discretised input  Δ * B_ssm * x

    Returns:
        h : (B, N, d)  — hidden state at every position (h_0 … h_{N-1})
                         where h_t = A_t * h_{t-1} + B_t,  h_{-1} = 0
    """
    B_sz, N, d = A.shape

    # Pad to next power of two for the binary tree
    L = 1 << math.ceil(math.log2(max(N, 1)))
    pad = L - N
    if pad > 0:
        pad_A = torch.ones(B_sz, pad, d, device=A.device, dtype=A.dtype)
        pad_B = torch.zeros(B_sz, pad, d, device=A.device, dtype=A.dtype)
        A = torch.cat([A, pad_A], dim=1)   # identity map for padding
        B = torch.cat([B, pad_B], dim=1)

    # --- Up-sweep: build the reduction tree --------------------------------
    # Store all levels so we can down-sweep later.
    # Level 0 is the original leaf layer.
    pa = [A]   # pa[k] has shape (B, L >> k, d)
    pb = [B]

    length = L
    while length > 1:
        length //= 2
        a_prev, b_prev = pa[-1], pb[-1]
        # Even indices are "left children", odd are "right children"
        a_left  = a_prev[:, 0::2, :]   # (B, length, d)
        b_left  = b_prev[:, 0::2, :]
        a_right = a_prev[:, 1::2, :]
        b_right = b_prev[:, 1::2, :]
        a_new, b_new = _compose(a_right, b_right, a_left, b_left)
        pa.append(a_new)
        pb.append(b_new)

    # --- Down-sweep: distribute prefix values back to every leaf -----------
    # Initialise the "carry" at the root with the identity map (a=1, b=0).
    carry_a = torch.ones(B_sz, 1, d, device=A.device, dtype=A.dtype)
    carry_b = torch.zeros(B_sz, 1, d, device=A.device, dtype=A.dtype)

    levels = len(pa) - 1   # number of up-sweep levels (excluding leaf)
    # Walk from the coarsest internal level down to the leaf level
    for k in range(levels, 0, -1):          # k = levels … 1
        a_level = pa[k - 1]                 # shape (B, 2^k, d)
        b_level = pb[k - 1]
        n_nodes = a_level.shape[1]          # = 2^k

        # Expand carry to cover every pair of children
        # carry has shape (B, n_nodes//2, d) — one entry per parent
        carry_a_exp = carry_a.expand(-1, n_nodes // 2, -1)   # (B, n_nodes//2, d)
        carry_b_exp = carry_b.expand(-1, n_nodes // 2, -1)

        # Left child gets the carry directly
        new_left_a, new_left_b = carry_a_exp, carry_b_exp

        # Right child gets carry composed with the left sibling
        a_left_sib = a_level[:, 0::2, :]   # left sibling at this level
        b_left_sib = b_level[:, 0::2, :]
        new_right_a, new_right_b = _compose(
            a_left_sib, b_left_sib, carry_a_exp, carry_b_exp
        )

        # Interleave left and right back into position order
        new_a = torch.stack([new_left_a, new_right_a], dim=2)  # (B, n//2, 2, d)
        new_b = torch.stack([new_left_b, new_right_b], dim=2)
        carry_a = new_a.reshape(B_sz, n_nodes, d)
        carry_b = new_b.reshape(B_sz, n_nodes, d)

    # carry_a / carry_b now hold the *prefix before* each leaf.
    # Apply the leaf's own map to get the inclusive prefix (= h_t).
    h = carry_a * pa[0] + carry_b   # inclusive: h_t = prefix_a * A_t + prefix_b
    # The formula above gives  carry o (A_t, B_t)  which equals h_t.
    # More precisely: h_t = A_t * carry_b + B_t  doesn't work for multi-dim.
    # Correct inclusive step:  (a_out, b_out) = compose(A_t, B_t, carry_a, carry_b)
    #                          h_t = b_out  (since initial hidden state = 0)
    _, h = _compose(pa[0], pb[0], carry_a, carry_b)

    # Trim padding and return
    return h[:, :N, :]
