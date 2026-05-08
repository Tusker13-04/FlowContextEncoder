"""
FlowDataset — PyTorch Dataset for packet flow classification.

Supports:
  - Real CICFlowMeter / Zeek-exported CSV flow data.
  - Synthetic data generation for testing the pipeline.

Each sample is a tuple:
    (packets, context, mask, label)
    packets: (N, d_in)   — per-packet features
    context: (d_ctx,)    — RTT, jitter, loss_rate, throughput
    mask:    (N,)        — bool, True = real packet
    label:   int         — application class index
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import numpy as np


class FlowDataset(Dataset):
    """
    Generic flow dataset. Pass pre-computed tensors or use `from_synthetic()`.

    Args:
        packets_list : list of (n_i, d_in) float tensors  (variable length)
        contexts     : (M, d_ctx) float tensor
        labels       : (M,) int tensor
        max_len      : pad/truncate all flows to this length
    """

    def __init__(
        self,
        packets_list: List[torch.Tensor],
        contexts:     torch.Tensor,
        labels:       torch.Tensor,
        max_len:      int = 128,
    ):
        self.max_len  = max_len
        self.contexts = contexts
        self.labels   = labels

        # Pre-pad all flows to max_len
        self.packets  = []
        self.masks    = []
        for pkts in packets_list:
            n = min(len(pkts), max_len)
            pad = torch.zeros(max_len, pkts.shape[-1])
            pad[:n] = pkts[:n]
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:n] = True
            self.packets.append(pad)
            self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        return self.packets[idx], self.contexts[idx], self.masks[idx], self.labels[idx]

    @classmethod
    def from_synthetic(
        cls,
        n_classes:   int = 6,
        n_per_class: int = 100,
        max_len:     int = 128,
        d_in:        int = 8,
        d_ctx:       int = 4,
        seed:        int = 42,
    ) -> 'FlowDataset':
        """
        Generate synthetic labeled flows.
        Classes are linearly separated in feature space so training loss should
        decrease smoothly — useful for smoke-testing the pipeline.
        """
        rng = np.random.default_rng(seed)
        all_pkts, all_ctx, all_lbl = [], [], []

        class_centers = rng.standard_normal((n_classes, d_ctx)) * 2.0

        for cls_idx in range(n_classes):
            for _ in range(n_per_class):
                flow_len = int(rng.integers(16, max_len + 1))
                # Class-specific pattern in packet features
                base   = rng.standard_normal(d_in) * 0.5 + (cls_idx - n_classes / 2)
                pkts   = rng.standard_normal((flow_len, d_in)) * 0.3 + base
                ctx    = rng.standard_normal(d_ctx) * 0.2 + class_centers[cls_idx]
                all_pkts.append(torch.tensor(pkts, dtype=torch.float32))
                all_ctx.append(ctx)
                all_lbl.append(cls_idx)

        contexts = torch.tensor(np.array(all_ctx), dtype=torch.float32)
        labels   = torch.tensor(all_lbl, dtype=torch.long)
        return cls(all_pkts, contexts, labels, max_len)
