"""
Phase 0 Configuration.
All hyperparameters in one place — override for Phase 1 experiments.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    d_in:     int   = 8      # per-packet feature dim
    d_ctx:    int   = 4      # network context dim (RTT, jitter, loss, tput)
    d_model:  int   = 128    # internal dimension
    d_emb:    int   = 128    # output embedding dim
    n_layers: int   = 4      # number of Mamba blocks
    d_state:  int   = 16     # SSM state dimension
    max_len:  int   = 128    # maximum packets per flow
    dropout:  float = 0.1


@dataclass
class TrainConfig:
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    batch_size:    int   = 64
    epochs:        int   = 50
    temperature:   float = 0.07   # SupConLoss tau
    warmup_epochs: int   = 5
    grad_clip:     float = 1.0
    seed:          int   = 42
    log_every:     int   = 10     # steps


@dataclass
class DataConfig:
    # Synthetic (Phase 0 smoke test)
    n_classes:   int = 6
    n_per_class: int = 200

    # Real data paths (fill in for Phase 1)
    cesnet_quic_path:  Optional[str] = None  # path to CESNET-QUIC22 parquet files
    fiveg_path:        Optional[str] = None  # path to 5G traffic CSV
    max_len:           int = 128


@dataclass
class Phase0Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
