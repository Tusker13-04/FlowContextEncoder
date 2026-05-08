"""
Phase 0 — Synthetic data generator.

Generates a small but realistic labelled flow dataset from pure Python
(no pcap files required) for smoke-testing the full pipeline locally.

Traffic classes and their statistical profiles
-----------------------------------------------
  0 video_streaming   large packets, low IAT, low jitter, high throughput
  1 gaming            small packets, very low IAT, ultra-low jitter, low RTT
  2 voip              tiny packets, regular IAT, low jitter, low RTT
  3 bulk_transfer     large packets, variable IAT, high throughput
  4 browsing          mixed sizes, bursty IAT, medium RTT
  5 xr_streaming      large packets, very low IAT, low jitter, high throughput
                      (close to video_streaming — tests intra-class cohesion)
"""

from __future__ import annotations

import random
import time

from .features import derive_context_from_packets

# ── Class profiles ─────────────────────────────────────────────────────────────

_PROFILES: dict[str, dict] = {
    "video_streaming": {
        "size_mu": 1200, "size_sig": 300,
        "iat_mu":  0.005, "iat_sig": 0.002,
        "rtt_ms":  30,    "jitter_ms": 3,  "throughput": 4_000_000,
    },
    "gaming": {
        "size_mu": 120,  "size_sig": 40,
        "iat_mu":  0.02,  "iat_sig": 0.003,
        "rtt_ms":  15,    "jitter_ms": 1,  "throughput": 200_000,
    },
    "voip": {
        "size_mu": 200,  "size_sig": 30,
        "iat_mu":  0.02,  "iat_sig": 0.001,
        "rtt_ms":  20,    "jitter_ms": 2,  "throughput": 300_000,
    },
    "bulk_transfer": {
        "size_mu": 1400, "size_sig": 100,
        "iat_mu":  0.001, "iat_sig": 0.002,
        "rtt_ms":  50,    "jitter_ms": 10, "throughput": 10_000_000,
    },
    "browsing": {
        "size_mu": 600,  "size_sig": 400,
        "iat_mu":  0.05,  "iat_sig": 0.05,
        "rtt_ms":  60,    "jitter_ms": 15, "throughput": 500_000,
    },
    "xr_streaming": {
        "size_mu": 1100, "size_sig": 250,
        "iat_mu":  0.004, "iat_sig": 0.002,
        "rtt_ms":  25,    "jitter_ms": 2,  "throughput": 5_000_000,
    },
}

CLASS_NAMES: list[str] = list(_PROFILES.keys())


def _make_flow(
    class_name: str,
    n_pkts: int,
    rng: random.Random,
) -> dict:
    """Generate one synthetic flow record."""
    p = _PROFILES[class_name]
    ts = time.time()
    pkts: list[dict] = []
    for _ in range(n_pkts):
        size = max(40, int(rng.gauss(p["size_mu"], p["size_sig"])))
        iat  = max(0.0, rng.gauss(p["iat_mu"],  p["iat_sig"]))
        ts  += iat
        pkts.append({
            "direction":  rng.choice([1, -1]),
            "size":       size,
            "timestamp":  ts,
            "tcp_flags":  0x10,  # ACK only
        })

    meta = {
        "rtt_ms":        p["rtt_ms"]    + rng.gauss(0, p["jitter_ms"]),
        "jitter_ms":     p["jitter_ms"] + abs(rng.gauss(0, 1)),
        "pkt_loss_rate": max(0.0, rng.gauss(0.01, 0.005)),
        "throughput":    p["throughput"] * rng.uniform(0.8, 1.2),
    }
    return {"pkts": pkts, "meta": meta, "label": CLASS_NAMES.index(class_name)}


def make_synthetic_dataset(
    n_per_class:  int  = 200,
    min_packets:  int  = 16,
    max_packets:  int  = 96,
    seed:         int  = 42,
) -> list[dict]:
    """
    Build a labelled list of raw flow dicts suitable for FlowDataset.

    Parameters
    ----------
    n_per_class : number of flows per traffic class
    min_packets : minimum packets per generated flow
    max_packets : maximum packets per generated flow
    seed        : reproducibility seed

    Returns
    -------
    list of dicts: {pkts, meta, label}
    """
    rng     = random.Random(seed)
    records = []
    for cls in CLASS_NAMES:
        for _ in range(n_per_class):
            n_pkts = rng.randint(min_packets, max_packets)
            records.append(_make_flow(cls, n_pkts, rng))
    rng.shuffle(records)
    return records
