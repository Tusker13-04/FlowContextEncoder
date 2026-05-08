"""
Phase 0 — FlowDataset and collate_flows.

FlowDataset accepts three interchange formats:

  1. List of raw-packet flows  (list[dict])   — richest, uses FlowFeatureExtractor
  2. Pre-extracted tensors     (list[tuple])  — (packets_tensor, ctx_tensor, label)
  3. CSV path                  (str | Path)   — one row per flow, columns described below

CSV schema (minimum required columns)
  label          : int   class id
  rtt_ms         : float
  jitter_ms      : float
  pkt_loss_rate  : float
  throughput     : float
  pkt_sizes      : str   semicolon-separated ints, e.g. "60;1460;800"
  pkt_dirs       : str   semicolon-separated ints (+1/-1)
  pkt_times      : str   semicolon-separated floats (absolute seconds)
  pkt_flags      : str   semicolon-separated ints (tcp flag bitmask) [optional]
"""

from __future__ import annotations

from pathlib import Path
from typing  import Sequence

import torch
from torch.utils.data import Dataset

from .features import FlowFeatureExtractor, derive_context_from_packets


class FlowDataset(Dataset):
    """
    Parameters
    ----------
    source : str | Path | list
        - str/Path → CSV file
        - list of (packets_tensor, ctx_tensor, label)  → pre-extracted
        - list of {'pkts': [...], 'meta': {...}, 'label': int} → raw
    max_packets : int
        Truncate / filter flows (default 128).
    min_packets : int
        Skip flows shorter than this (default 4).
    label_map : dict[str, int] | None
        Optional string→int label mapping (applied when reading CSV with
        string labels).
    """

    def __init__(
        self,
        source,
        max_packets: int = 128,
        min_packets: int = 4,
        label_map: dict | None = None,
    ):
        self.extractor  = FlowFeatureExtractor(max_packets, min_packets)
        self.label_map  = label_map or {}
        self._samples: list[tuple[torch.Tensor, torch.Tensor, int]] = []

        if isinstance(source, (str, Path)):
            self._load_csv(Path(source))
        elif isinstance(source, (list, tuple)) and len(source) > 0:
            first = source[0]
            if isinstance(first, dict):
                self._load_raw(source)
            elif isinstance(first, (list, tuple)) and len(first) == 3:
                self._load_pretensored(source)
            else:
                raise ValueError(
                    "source list must contain dicts (raw) or "
                    "3-tuples (packets, ctx, label)."
                )
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        packets, ctx, label = self._samples[idx]
        return packets, ctx, label

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_raw(self, records: list[dict]) -> None:
        for rec in records:
            pkts  = rec["pkts"]
            meta  = rec.get("meta") or derive_context_from_packets(pkts)
            label = self._resolve_label(rec.get("label", 0))
            if len(pkts) < self.extractor.min_packets:
                continue
            packets_t, ctx_t = self.extractor.extract(pkts, meta)
            self._samples.append((packets_t, ctx_t, label))

    def _load_pretensored(
        self, records: Sequence[tuple[torch.Tensor, torch.Tensor, int]]
    ) -> None:
        for packets, ctx, label in records:
            self._samples.append(
                (packets.float(), ctx.float(), int(label))
            )

    def _load_csv(self, path: Path) -> None:
        import csv

        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = self._resolve_label(row.get("label", "0"))

                sizes  = _parse_seq(row.get("pkt_sizes",  ""), float)
                dirs   = _parse_seq(row.get("pkt_dirs",   ""), float)
                times  = _parse_seq(row.get("pkt_times",  ""), float)
                flags  = _parse_seq(row.get("pkt_flags",  ""), int)

                n = min(len(sizes), len(dirs), len(times))
                if n < self.extractor.min_packets:
                    continue

                pkts = [
                    {
                        "direction":  dirs[i] if i < len(dirs) else 1,
                        "size":       sizes[i],
                        "timestamp":  times[i],
                        "tcp_flags":  flags[i] if i < len(flags) else 0,
                    }
                    for i in range(n)
                ]

                meta = {
                    "rtt_ms":        float(row.get("rtt_ms",        0)),
                    "jitter_ms":     float(row.get("jitter_ms",     0)),
                    "pkt_loss_rate": float(row.get("pkt_loss_rate", 0)),
                    "throughput":    float(row.get("throughput",    0)),
                }
                # If context cols are all zero, heuristically derive them
                if all(v == 0.0 for v in meta.values()):
                    meta = derive_context_from_packets(pkts)

                packets_t, ctx_t = self.extractor.extract(pkts, meta)
                self._samples.append((packets_t, ctx_t, label))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_label(self, raw) -> int:
        if isinstance(raw, int):
            return raw
        s = str(raw).strip()
        if s in self.label_map:
            return self.label_map[s]
        try:
            return int(s)
        except ValueError:
            # Auto-assign: string label → next available int
            idx = len(self.label_map)
            self.label_map[s] = idx
            return idx

    @property
    def num_classes(self) -> int:
        return len({label for _, _, label in self._samples})

    @property
    def label_names(self) -> dict[int, str]:
        return {v: k for k, v in self.label_map.items()}


# ── Collate ────────────────────────────────────────────────────────────────────

def collate_flows(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Variable-length flow collation with zero-padding and boolean mask.

    Returns
    -------
    packets : (B, N_max, PACKET_FEAT_DIM)   float32, zero-padded
    ctx     : (B, CTX_FEAT_DIM)             float32
    mask    : (B, N_max)                    bool   (True = real packet)
    labels  : (B,)                          int64
    """
    packets_list, ctx_list, labels = zip(*batch)

    n_max   = max(p.shape[0] for p in packets_list)
    B       = len(packets_list)
    d_pkt   = packets_list[0].shape[1]

    padded  = torch.zeros(B, n_max, d_pkt, dtype=torch.float32)
    mask    = torch.zeros(B, n_max, dtype=torch.bool)

    for i, pkt in enumerate(packets_list):
        n = pkt.shape[0]
        padded[i, :n] = pkt
        mask[i, :n]   = True

    ctx_batch    = torch.stack([c.float() for c in ctx_list],  dim=0)  # (B, d_ctx)
    label_batch  = torch.tensor(labels, dtype=torch.int64)              # (B,)

    return padded, ctx_batch, mask, label_batch


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_seq(s: str, typ):
    """Parse a semicolon-separated string into a typed list."""
    if not s.strip():
        return []
    return [typ(x) for x in s.split(";") if x.strip()]
