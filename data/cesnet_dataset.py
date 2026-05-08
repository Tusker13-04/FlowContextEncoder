"""
cesnet_dataset.py  —  Streaming loader for CESNET-QUIC22 via cesnet-datazoo

Requires
--------
  pip install cesnet-datazoo  # https://pypi.org/project/cesnet-datazoo/

CESNET app_id → coarse class mapping
--------------------------------------
  The CESNET-QUIC22 dataset has 102 application classes identified by app_id
  integers mapped to SNI / service names.  We collapse them into 5 coarse
  classes matching the problem statement.

Usage
-----
  python data/cesnet_dataset.py \
      --data-root ~/.cesnet \
      --out      data/flows/cesnet_quic22.parquet \
      --max-flows 500000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd


N_MAX = 128

# ──────────────────────────────────────────────────────────────────────────────
# CESNET app_id / service name → coarse class
# Based on CESNET-QUIC22 documentation:
# https://cesnet.github.io/cesnet-datazoo/
# ──────────────────────────────────────────────────────────────────────────────
CESNET_COARSE: Dict[str, str] = {
    # Video streaming
    "youtube":           "video_streaming",
    "netflix":           "video_streaming",
    "twitch":            "video_streaming",
    "disneyplus":       