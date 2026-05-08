"""
Phase 0 — Feature extraction.

Converts a raw flow (list of packet dicts) into two tensors:

  packets : (N, PACKET_FEAT_DIM)  — per-packet feature matrix
  ctx     : (CTX_FEAT_DIM,)       — per-flow context vector

Packet features  (6-dim, PACKET_FEAT_DIM = 6)
  0  direction           +1 = client→server, -1 = server→client
  1  packet_size         bytes, log1p-normalised
  2  inter_arrival_time  seconds since previous packet in this flow, log1p-normalised
  3  tcp_flag_syn        0/1
  4  tcp_flag_ack        0/1
  5  tcp_flag_fin        0/1

Context features  (4-dim, CTX_FEAT_DIM = 4)
  0  rtt_ms        round-trip time in milliseconds, log1p-normalised
  1  jitter_ms     jitter in ms, log1p-normalised
  2  pkt_loss_rate fraction of retransmitted packets [0, 1]
  3  throughput    bytes/second, log1p-normalised

All features are float32.  Unknown / missing values are filled with 0.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# ── Public constants (consumed by model.__init__ as d_in / d_ctx) ────────────
PACKET_FEAT_DIM: int = 6
CTX_FEAT_DIM: int    = 4


class FlowFeatureExtractor:
    """
    Stateless extractor — no fitting required.

    Parameters
    ----------
    max_packets : int
        Truncate flows longer than this (default 128).
    min_packets : int
        Flows shorter than this are skipped during dataset construction.
    """

    def __init__(self, max_packets: int = 128, min_packets: int = 4):
        self.max_packets = max_packets
        self.min_packets = min_packets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        pkts: list[dict[str, Any]],
        flow_meta: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pkts : list of dicts, each with keys:
            'direction'   : int   +1 | -1
            'size'        : int   bytes
            'timestamp'   : float seconds (absolute)
            'tcp_flags'   : int   bitmask  (SYN=0x02, ACK=0x10, FIN=0x01)  [optional]

        flow_meta : dict with optional keys:
            'rtt_ms'        : float
            'jitter_ms'     : float
            'pkt_loss_rate' : float   [0, 1]
            'throughput'    : float   bytes/s

        Returns
        -------
        packets : torch.Tensor  (N, PACKET_FEAT_DIM)  float32
        ctx     : torch.Tensor  (CTX_FEAT_DIM,)       float32
        """
        pkts = pkts[: self.max_packets]
        n    = len(pkts)

        rows = torch.zeros(n, PACKET_FEAT_DIM, dtype=torch.float32)
        prev_ts: float | None = None

        for i, p in enumerate(pkts):
            rows[i, 0] = float(p.get("direction", 1))
            rows[i, 1] = math.log1p(float(p.get("size", 0)))

            ts = float(p.get("timestamp", 0.0))
            if prev_ts is not None:
                rows[i, 2] = math.log1p(max(ts - prev_ts, 0.0))
            prev_ts = ts

            flags = int(p.get("tcp_flags", 0))
            rows[i, 3] = float(bool(flags & 0x02))  # SYN
            rows[i, 4] = float(bool(flags & 0x10))  # ACK
            rows[i, 5] = float(bool(flags & 0x01))  # FIN

        ctx = self._extract_ctx(flow_meta or {})
        return rows, ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ctx(meta: dict[str, Any]) -> torch.Tensor:
        c = torch.zeros(CTX_FEAT_DIM, dtype=torch.float32)
        c[0] = math.log1p(float(meta.get("rtt_ms",        0.0)))
        c[1] = math.log1p(float(meta.get("jitter_ms",     0.0)))
        c[2] = float(meta.get("pkt_loss_rate", 0.0))
        c[3] = math.log1p(float(meta.get("throughput",    0.0)))
        return c


# ── Convenience: derive context from packet list alone ────────────────────────

def derive_context_from_packets(
    pkts: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Heuristically estimate RTT, jitter, loss, and throughput from raw
    packet timestamps and sizes alone — useful when Zeek conn.log is
    unavailable (e.g., working from plain pcap).

    Estimates
    ---------
    rtt_ms        : 2 × median inter-arrival time of SYN→SYN/ACK pairs
                    (falls back to 2 × median forward IAT)
    jitter_ms     : std-dev of all inter-arrival times
    pkt_loss_rate : fraction of RTX-flagged packets (tcp_flags has PSH+ACK
                    without prior data ACK — rough proxy; 0 if unavailable)
    throughput    : total bytes / flow duration in seconds
    """
    import statistics

    if not pkts:
        return {"rtt_ms": 0.0, "jitter_ms": 0.0, "pkt_loss_rate": 0.0, "throughput": 0.0}

    timestamps = [float(p.get("timestamp", 0.0)) for p in pkts]
    sizes      = [float(p.get("size",      0))   for p in pkts]

    iats: list[float] = []
    for i in range(1, len(timestamps)):
        iat = (timestamps[i] - timestamps[i - 1]) * 1000.0  # → ms
        if iat >= 0:
            iats.append(iat)

    rtt_ms    = 2.0 * statistics.median(iats) if iats else 0.0
    jitter_ms = statistics.stdev(iats)         if len(iats) > 1 else 0.0

    duration   = max(timestamps[-1] - timestamps[0], 1e-6)
    throughput = sum(sizes) / duration  # bytes/s

    return {
        "rtt_ms":        rtt_ms,
        "jitter_ms":     jitter_ms,
        "pkt_loss_rate": 0.0,     # cannot derive from IAT alone
        "throughput":    throughput,
    }
