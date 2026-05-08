"""
data/cesnet_dataset.py
======================
Streaming loader for CESNET-QUIC22 (and other CESNET DataZoo datasets)
using the cesnet-datazoo library.  Maps fine-grained CESNET application
IDs to the five coarse traffic classes used throughout this project.

Requires
--------
    pip install cesnet-datazoo>=0.8.0

Usage
-----
    from data.cesnet_dataset import CESNETFlowDataset, stream_to_parquet

    # Stream and save:
    stream_to_parquet(
        dataset_name="CESNET-QUIC22",
        output_dir="data/flows",
        max_flows=500_000,
    )

    # Or use directly as an iterable:
    for batch in CESNETFlowDataset("CESNET-QUIC22", split="train", batch_size=1024):
        packets, context, labels = batch   # torch.Tensors
"""

import warnings
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

# ── DataZoo import (optional at import time) ──────────────────────────────────
try:
    from cesnet_datazoo.datasets import CESNET_QUIC22, CESNET_TLS22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    DATAZOO_AVAILABLE = True
except ImportError:
    DATAZOO_AVAILABLE = False
    warnings.warn(
        "cesnet-datazoo not installed. Run: pip install cesnet-datazoo\n"
        "CESNET dataset loaders will not function until it is installed."
    )

# ── Coarse class mapping ──────────────────────────────────────────────────────
# Map CESNET application/category strings → project coarse classes.
# Extend this dict as you inspect actual CESNET app IDs.
_CESNET_TO_COARSE = {
    # Video streaming
    "youtube": "video_streaming",
    "netflix": "video_streaming",
    "twitch": "video_streaming",
    "primevideo": "video_streaming",
    "disneyplus": "video_streaming",
    "vimeo": "video_streaming",
    "video": "video_streaming",
    "streaming": "video_streaming",
    # Gaming
    "steam": "gaming",
    "riotgames": "gaming",
    "epicgames": "gaming",
    "battlenet": "gaming",
    "game": "gaming",
    "gaming": "gaming",
    # VoIP / real-time comms
    "zoom": "voip",
    "teams": "voip",
    "skype": "voip",
    "webex": "voip",
    "discord": "voip",
    "voip": "voip",
    "meet": "voip",
    # Web / general HTTPS
    "google": "web",
    "facebook": "web",
    "twitter": "web",
    "instagram": "web",
    "web": "web",
    "http": "web",
    "quic": "web",
    # XR / emerging
    "xr": "xr",
    "vr": "xr",
    "ar": "xr",
}

SEQ_LEN = 128
COARSE_CLASSES = ["video_streaming", "gaming", "voip", "web", "xr"]
CLASS_TO_IDX = {c: i for i, c in enumerate(COARSE_CLASSES)}


def _map_app_to_coarse(app_name: str) -> Optional[str]:
    """Map a CESNET app label to a coarse class. Returns None if unmapped."""
    name_lower = str(app_name).lower()
    for key, coarse in _CESNET_TO_COARSE.items():
        if key in name_lower:
            return coarse
    return None  # 'background' or unrecognised → skip


def _cesnet_row_to_flow(row, coarse_label: str, split: str) -> Optional[dict]:
    """
    Convert a single CESNET DataZoo row dict to our standard flow dict.
    DataZoo rows expose packet features as numpy arrays under keys like
    'PPI_IPT', 'PPI_PKT_LEN', 'PPI_FLAGS', etc.
    """
    try:
        pkt_len = np.asarray(row.get("PPI_PKT_LEN", []), dtype=np.float32)
        pkt_ipt = np.asarray(row.get("PPI_IPT", []), dtype=np.float32)   # inter-packet time ms
        pkt_dir = np.asarray(row.get("PPI_DIR", []), dtype=np.float32)
        pkt_flags = np.asarray(row.get("PPI_FLAGS", []), dtype=np.float32)

        n = min(len(pkt_len), SEQ_LEN)
        if n < 4:
            return None

        arr = np.zeros((SEQ_LEN, 5), dtype=np.float32)
        arr[:n, 0] = np.clip(pkt_len[:n] / 1500.0, 0, 1)
        arr[:n, 1] = np.clip(pkt_dir[:n], 0, 1) if len(pkt_dir) >= n else 0
        arr[:n, 2] = np.log1p(pkt_ipt[:n]) if len(pkt_ipt) >= n else 0
        arr[:n, 3] = (pkt_flags[:n] / 63.0) if len(pkt_flags) >= n else 0
        # QUIC: CESNET-QUIC22 flows are all QUIC by definition
        arr[:n, 4] = 1.0

        rtt_ms = float(row.get("QUIC_RTT", row.get("TLS_RTT", np.nan)))
        jitter_ms = float(np.std(pkt_ipt[:n])) if n > 1 else 0.0

        return {
            "flow_id": str(row.get("FLOW_ID", id(row))),
            "app_label": coarse_label,
            "packets": arr,
            "rtt_ms": rtt_ms,
            "jitter_ms": jitter_ms,
            "pkt_loss_rate": 0.0,
            "throughput_kbps": float(row.get("BYTES", 0) * 8 / 1000 / max(float(row.get("DURATION", 1)), 1e-6)),
            "split": split,
        }
    except Exception:
        return None


class CESNETFlowDataset:
    """
    Iterable dataset over CESNET-QUIC22 or CESNET-TLS22.
    Yields batches of (packets_tensor, context_tensor, label_tensor).

    Parameters
    ----------
    dataset_name : "CESNET-QUIC22" | "CESNET-TLS22"
    split        : "train" | "val" | "test"
    data_root    : local path where DataZoo stores downloaded data
    batch_size   : number of flows per batch
    max_flows    : stop after this many flows (None = all)
    """

    def __init__(
        self,
        dataset_name: str = "CESNET-QUIC22",
        split: str = "train",
        data_root: str = "data/cesnet",
        batch_size: int = 256,
        max_flows: Optional[int] = None,
    ):
        if not DATAZOO_AVAILABLE:
            raise ImportError("cesnet-datazoo is required. Run: pip install cesnet-datazoo")

        self.split = split
        self.batch_size = batch_size
        self.max_flows = max_flows

        DatasetClass = CESNET_QUIC22 if "QUIC" in dataset_name.upper() else CESNET_TLS22
        cfg = DatasetConfig(
            data_root=data_root,
            train_period_name=split if split == "train" else None,
            val_period_name=split if split == "val" else None,
            test_period_name=split if split == "test" else None,
        )
        self._dataset = DatasetClass(cfg)
        print(f"[CESNETFlowDataset] Loaded {dataset_name} split={split}")

    def __iter__(self) -> Iterator[Tuple]:
        import torch
        batch_pkts, batch_ctx, batch_labels = [], [], []
        count = 0

        for row in self._dataset:
            app_name = row.get("APP", row.get("CATEGORY", "unknown"))
            coarse = _map_app_to_coarse(str(app_name))
            if coarse is None:
                continue

            flow = _cesnet_row_to_flow(row, coarse, self.split)
            if flow is None:
                continue

            batch_pkts.append(flow["packets"])
            batch_ctx.append([
                flow["rtt_ms"] if not np.isnan(flow["rtt_ms"]) else 0.0,
                flow["jitter_ms"],
                flow["pkt_loss_rate"],
                flow["throughput_kbps"],
            ])
            batch_labels.append(CLASS_TO_IDX[coarse])

            if len(batch_pkts) == self.batch_size:
                yield (
                    torch.tensor(np.stack(batch_pkts), dtype=torch.float32),
                    torch.tensor(batch_ctx, dtype=torch.float32),
                    torch.tensor(batch_labels, dtype=torch.long),
                )
                batch_pkts, batch_ctx, batch_labels = [], [], []

            count += 1
            if self.max_flows and count >= self.max_flows:
                break

        if batch_pkts:
            yield (
                torch.tensor(np.stack(batch_pkts), dtype=torch.float32),
                torch.tensor(batch_ctx, dtype=torch.float32),
                torch.tensor(batch_labels, dtype=torch.long),
            )


def stream_to_parquet(
    dataset_name: str = "CESNET-QUIC22",
    output_dir: str = "data/flows",
    data_root: str = "data/cesnet",
    split: str = "train",
    max_flows: Optional[int] = 500_000,
    chunk_size: int = 10_000,
):
    """
    Stream CESNET dataset and write to parquet files in chunks.
    Safe for datasets with 100M+ flows — never loads all data into RAM.
    """
    if not DATAZOO_AVAILABLE:
        raise ImportError("cesnet-datazoo is required.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    DatasetClass = CESNET_QUIC22 if "QUIC" in dataset_name.upper() else CESNET_TLS22
    cfg = DatasetConfig(data_root=data_root)
    dataset = DatasetClass(cfg)

    rows, chunk_idx, total = [], 0, 0

    for row in dataset:
        app_name = row.get("APP", row.get("CATEGORY", "unknown"))
        coarse = _map_app_to_coarse(str(app_name))
        if coarse is None:
            continue

        flow = _cesnet_row_to_flow(row, coarse, split)
        if flow is None:
            continue

        rows.append(flow)
        total += 1

        if len(rows) >= chunk_size:
            path = out_dir / f"cesnet_{dataset_name.lower()}_{split}_chunk{chunk_idx:04d}.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            print(f"[stream_to_parquet] chunk {chunk_idx:04d} → {path}  (total: {total:,})")
            rows, chunk_idx = [], chunk_idx + 1

        if max_flows and total >= max_flows:
            break

    if rows:
        path = out_dir / f"cesnet_{dataset_name.lower()}_{split}_chunk{chunk_idx:04d}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"[stream_to_parquet] chunk {chunk_idx:04d} → {path}  (total: {total:,})")

    print(f"[stream_to_parquet] Done. Total flows written: {total:,}")
