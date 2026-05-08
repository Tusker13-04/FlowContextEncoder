"""
Phase 0 — build_loaders: train / val / test DataLoaders.

Usage
-----
    from data import build_loaders

    train_dl, val_dl, test_dl = build_loaders(
        source      = "data/flows.csv",
        batch_size  = 64,
        val_split   = 0.15,
        test_split  = 0.10,
        seed        = 42,
    )

    for packets, ctx, mask, labels in train_dl:
        # packets : (B, N, 6)   ctx : (B, 4)
        # mask    : (B, N) bool labels : (B,)
        ...
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split

from .dataset import FlowDataset, collate_flows


def build_loaders(
    source: Any,
    batch_size:  int   = 64,
    val_split:   float = 0.15,
    test_split:  float = 0.10,
    num_workers: int   = 0,
    seed:        int   = 42,
    max_packets: int   = 128,
    min_packets: int   = 4,
    label_map:   dict | None = None,
    pin_memory:  bool  = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from any source accepted by
    FlowDataset.

    Parameters
    ----------
    source      : CSV path, list of raw-packet dicts, or list of pre-tensored
                  (packets, ctx, label) tuples — forwarded to FlowDataset.
    batch_size  : samples per mini-batch.
    val_split   : fraction of data for validation (default 0.15).
    test_split  : fraction of data for testing    (default 0.10).
    num_workers : DataLoader worker processes (keep 0 on Windows).
    seed        : random seed for the split.
    max_packets : forwarded to FlowDataset.
    min_packets : forwarded to FlowDataset.
    label_map   : optional string→int label mapping.
    pin_memory  : set True when using CUDA.

    Returns
    -------
    train_dl, val_dl, test_dl : DataLoader (use collate_flows automatically)
    """
    ds = FlowDataset(
        source,
        max_packets = max_packets,
        min_packets = min_packets,
        label_map   = label_map,
    )

    total      = len(ds)
    n_test     = max(1, int(total * test_split))
    n_val      = max(1, int(total * val_split))
    n_train    = total - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Dataset has only {total} samples — too few to split with "
            f"val_split={val_split}, test_split={test_split}."
        )

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=gen
    )

    _collate = collate_flows

    train_dl = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = _collate,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        collate_fn  = _collate,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        collate_fn  = _collate,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    print(
        f"[build_loaders] {total} flows → "
        f"train={n_train}  val={n_val}  test={n_test}  "
        f"| {ds.num_classes} classes"
    )
    return train_dl, val_dl, test_dl
