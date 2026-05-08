"""
FlowDataset  — a PyTorch Dataset wrapper for pre-extracted flow feature arrays.

Expected data format  (one .npz file per split):
    packets  : float32  (M, N_max, D_IN)   — zero-padded packet sequences
    ctx      : float32  (M, D_CTX)          — per-flow context vectors
    labels   : int64    (M,)                — integer app-type labels
    lengths  : int64    (M,)                — true flow length before padding

collate_flows builds the variable-length batch with a boolean padding mask.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from .features import make_synthetic_flow, D_IN, D_CTX


class FlowDataset(Dataset):
    """
    Args:
        npz_path   : path to a .npz file with keys packets/ctx/labels/lengths.
                     If None, generates a synthetic dataset for smoke-testing.
        max_len    : truncate / pad all flows to this length
        n_synth    : if npz_path is None, number of synthetic flows to generate
        n_classes  : number of app-type classes (synthetic mode only)
        seed       : RNG seed for synthetic generation
    """

    def __init__(
        self,
        npz_path: Optional[str | Path] = None,
        max_len: int = 128,
        n_synth: int = 1000,
        n_classes: int = 5,
        seed: int = 42,
    ):
        self.max_len = max_len

        if npz_path is not None:
            data = np.load(npz_path)
            self.packets = data["packets"].astype(np.float32)
            self.ctx     = data["ctx"].astype(np.float32)
            self.labels  = data["labels"].astype(np.int64)
            self.lengths = data["lengths"].astype(np.int64)
        else:
            self._build_synthetic(n_synth, n_classes, max_len, seed)

    def _build_synthetic(
        self,
        n_synth: int,
        n_classes: int,
        max_len: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        pkt_list, ctx_list, lbl_list, len_list = [], [], [], []

        for i in range(n_synth):
            n_pkts = int(rng.integers(16, max_len + 1))
            feats, ctx, label = make_synthetic_flow(
                app_type=i % n_classes,
                n_packets=n_pkts,
                rng=rng,
            )
            # Pad / truncate to max_len
            pad = max_len - n_pkts
            if pad > 0:
                feats = np.pad(feats, ((0, pad), (0, 0)))
            else:
                feats = feats[:max_len]
            pkt_list.append(feats)
            ctx_list.append(ctx)
            lbl_list.append(label)
            len_list.append(min(n_pkts, max_len))

        self.packets = np.stack(pkt_list).astype(np.float32)
        self.ctx     = np.stack(ctx_list).astype(np.float32)
        self.labels  = np.array(lbl_list, dtype=np.int64)
        self.lengths = np.array(len_list, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.packets[idx]),
            torch.from_numpy(self.ctx[idx]),
            torch.tensor(self.labels[idx]),
            torch.tensor(self.lengths[idx]),
        )

    def save(self, path: str | Path) -> None:
        """Save dataset to .npz for reuse."""
        np.savez_compressed(
            path,
            packets=self.packets,
            ctx=self.ctx,
            labels=self.labels,
            lengths=self.lengths,
        )


def collate_flows(batch):
    """
    Custom collate function for variable-length flows.
    Builds a boolean padding mask  (B, N)  where True = real packet.
    """
    packets, ctx, labels, lengths = zip(*batch)
    packets = torch.stack(packets)    # (B, N_max, D_IN)
    ctx     = torch.stack(ctx)        # (B, D_CTX)
    labels  = torch.stack(labels)     # (B,)
    lengths = torch.stack(lengths)    # (B,)

    N = packets.shape[1]
    mask = torch.arange(N).unsqueeze(0) < lengths.unsqueeze(1)  # (B, N) bool

    return packets, ctx, labels, mask
