"""
Feature extraction helpers.

Per-packet features (d_in = 6):
  [0] direction        : 0 = client→server, 1 = server→client
  [1] pkt_size_norm    : packet size / 1500  (normalised by max Ethernet MTU)
  [2] iat_log          : log1p(inter-arrival time in ms)
  [3] tcp_flag_bits    : packed TCP flags / 64  (0..1)
  [4] is_quic          : 1 if QUIC, 0 otherwise
  [5] position_frac    : t / (N-1)  — relative position in the flow

Flow-level context vector (d_ctx = 4):
  [0] rtt_ms_norm      : RTT estimate in ms / 500
  [1] jitter_norm      : jitter in ms / 100
  [2] retransmit_rate  : retransmissions / total packets  (0..1)
  [3] pkt_rate_norm    : packets per second / 1000
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any


D_IN  = 6   # per-packet feature dimension
D_CTX = 4   # context feature dimension


def extract_flow_features(
    packets: list[Dict[str, Any]],
    rtt_ms: float = 20.0,
    jitter_ms: float = 2.0,
    retransmit_rate: float = 0.0,
    pkt_rate: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-packet feature matrix and flow context vector from a list of
    packet dicts.  Each dict should have:
        direction  : int   (0 or 1)
        size       : int   (bytes)
        iat_ms     : float (inter-arrival time in ms, 0 for first packet)
        tcp_flags  : int   (0..63 — 6 TCP flag bits packed)
        is_quic    : bool

    Returns:
        pkt_feats : np.ndarray  (N, D_IN)   float32
        ctx_feats : np.ndarray  (D_CTX,)    float32
    """
    N = len(packets)
    pkt_feats = np.zeros((N, D_IN), dtype=np.float32)

    for t, pkt in enumerate(packets):
        pkt_feats[t, 0] = float(pkt.get("direction", 0))
        pkt_feats[t, 1] = min(pkt.get("size", 0) / 1500.0, 1.0)
        pkt_feats[t, 2] = float(np.log1p(max(pkt.get("iat_ms", 0.0), 0.0)))
        pkt_feats[t, 3] = min(pkt.get("tcp_flags", 0) / 64.0, 1.0)
        pkt_feats[t, 4] = float(pkt.get("is_quic", False))
        pkt_feats[t, 5] = t / max(N - 1, 1)

    ctx_feats = np.array([
        min(rtt_ms / 500.0, 1.0),
        min(jitter_ms / 100.0, 1.0),
        min(retransmit_rate, 1.0),
        min(pkt_rate / 1000.0, 1.0),
    ], dtype=np.float32)

    return pkt_feats, ctx_feats


def make_synthetic_flow(
    app_type: int,
    n_packets: int = 64,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a synthetic flow for unit-testing and smoke-tests.

    App type statistics (rough approximation):
        0  video streaming  — large packets, moderate IAT, low jitter
        1  gaming           — small packets, low IAT, variable jitter
        2  VoIP             — tiny packets, very regular IAT, low jitter
        3  bulk transfer    — max-size packets, bursty IAT
        4  XR / immersive   — mix of small control + large media packets
    """
    if rng is None:
        rng = np.random.default_rng()

    profiles = {
        0: dict(size_mu=1200, size_std=200, iat_mu=5,  iat_std=1,  jitter=2,  rtt=30),
        1: dict(size_mu=120,  size_std=60,  iat_mu=16, iat_std=8,  jitter=10, rtt=20),
        2: dict(size_mu=160,  size_std=20,  iat_mu=20, iat_std=1,  jitter=1,  rtt=15),
        3: dict(size_mu=1450, size_std=50,  iat_mu=1,  iat_std=5,  jitter=3,  rtt=40),
        4: dict(size_mu=600,  size_std=400, iat_mu=8,  iat_std=4,  jitter=8,  rtt=25),
    }
    p = profiles.get(app_type % 5, profiles[0])

    packets = []
    for t in range(n_packets):
        packets.append({
            "direction":  int(rng.integers(0, 2)),
            "size":       int(np.clip(rng.normal(p["size_mu"], p["size_std"]), 40, 1500)),
            "iat_ms":     float(np.clip(rng.normal(p["iat_mu"], p["iat_std"]), 0.0, 500.0)),
            "tcp_flags":  int(rng.integers(0, 64)),
            "is_quic":    bool(rng.random() < 0.3),
        })

    rtt    = p["rtt"]  + rng.normal(0, 3)
    jitter = p["jitter"] + rng.normal(0, 1)
    pkt_rate = n_packets / max(sum(pk["iat_ms"] for pk in packets) / 1000.0, 0.01)

    feats, ctx = extract_flow_features(
        packets,
        rtt_ms=float(np.clip(rtt, 1, 500)),
        jitter_ms=float(np.clip(jitter, 0, 100)),
        pkt_rate=float(np.clip(pkt_rate, 1, 1000)),
    )
    return feats, ctx, app_type
