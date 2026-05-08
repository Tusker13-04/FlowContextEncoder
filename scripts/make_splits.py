"""
make_splits.py  —  Leakage-resistant train / val / test / few-shot splits

Two split strategies
---------------------
  1. time  : train on weeks 1-3, val on week 4, test on week 5  (default for CESNET)
  2. random: stratified random split with no temporal guarantee

Few-shot split
--------------
  3 app classes are held out entirely from train/val/test and saved to
  splits/fewshot_apps.txt.  Their flow IDs go to splits/fewshot.txt.

Output
------
  splits/train.txt      — flow IDs, one per line
  splits/val.txt
  splits/test.txt
  splits/fewshot.txt
  splits/fewshot_apps.txt  — held-out class names

Usage
-----
  python scripts/make_splits.py \
      --parquet data/flows/cesnet.parquet \
      --strategy time \
      --ts-col ts_start          # column holding epoch seconds

  python scripts/make_splits.py \
      --parquet data/flows/lab.parquet \
      --strategy random
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


SPLITS_DIR = Path("splits")
FEWSHOT_N_CLASSES = 3


def _write_ids(path: Path, ids: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")
    print(f"[make_splits] {path}  ({len(ids)} flows)")


def time_split(df: pd.DataFrame, ts_col: str) -> dict:
    """
    Splits by quantile of timestamp — approximates week-based splits
    without requiring exact calendar metadata.
      train : ts < 60th percentile
      val   : 60th <= ts < 80th percentile
      test  : ts >= 80th percentile
    If ts_col not in df, falls back to row-order split.
    """
    if ts_col not in df.columns:
        print(f"[make_splits] WARNING: '{ts_col}' not found; falling back to row-order split.")
        n = len(df)
        idx = df.index.tolist()
        return {
            "train": df.loc[idx[:int(n * 0.6)], "flow_id"].tolist(),
            "val":   df.loc[idx[int(n * 0.6):int(n * 0.8)], "flow_id"].tolist(),
            "test":  df.loc[idx[int(n * 0.8):], "flow_id"].tolist(),
        }

    q60 = df[ts_col].quantile(0.60)
    q80 = df[ts_col].quantile(0.80)
    return {
        "train": df[df[ts_col] <  q60]["flow_id"].tolist(),
        "val":   df[(df[ts_col] >= q60) & (df[ts_col] < q80)]["flow_id"].tolist(),
        "test":  df[df[ts_col] >= q80]["flow_id"].tolist(),
    }


def random_split(df: pd.DataFrame, seed: int = 42) -> dict:
    """Stratified random split preserving class proportions."""
    rng = random.Random(seed)
    train_ids, val_ids, test_ids = [], [], []
    for label, group in df.groupby("app_label"):
        ids = group["flow_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        tr = int(n * 0.70)
        va = int(n * 0.85)
        train_ids.extend(ids[:tr])
        val_ids.extend(ids[tr:va])
        test_ids.extend(ids[va:])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def carve_fewshot(df: pd.DataFrame, split_ids: dict, n_classes: int = FEWSHOT_N_CLASSES, seed: int = 0) -> tuple:
    """
    Selects `n_classes` app labels to hold out entirely as few-shot eval classes.
    Removes their flow IDs from train/val/test and returns them as fewshot split.
    Chooses the least-frequent classes so training is affected minimally.
    """
    class_counts = df["app_label"].value_counts()
    all_classes  = class_counts.index.tolist()
    if len(all_classes) <= n_classes + 2:
        n_classes = max(1, len(all_classes) - 2)
    rng = random.Random(seed)
    # prefer smaller classes for held-out (less disruption to training)
    candidate_pool = all_classes[-(n_classes * 2):]
    held_out = rng.sample(candidate_pool, n_classes)

    held_set = set(df[df["app_label"].isin(held_out)]["flow_id"].tolist())
    fewshot_ids = list(held_set)

    for k in split_ids:
        split_ids[k] = [fid for fid in split_ids[k] if fid not in held_set]

    return split_ids, fewshot_ids, held_out


def main():
    parser = argparse.ArgumentParser(description="Generate leakage-resistant flow splits")
    parser.add_argument("--parquet",  required=True,       help="Input Parquet with flow_id and app_label columns")
    parser.add_argument("--strategy", default="time",       choices=["time", "random"])
    parser.add_argument("--ts-col",   default="ts_start",   help="Timestamp column for time-based split")
    parser.add_argument("--out-dir",  default="splits",     help="Output directory")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no-fewshot", action="store_true", help="Skip few-shot class carving")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[make_splits] Loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet, columns=[c for c in
        ["flow_id", "app_label", args.ts_col] if True])
    # drop ts_col if missing
    df = df[[c for c in df.columns if c in ["flow_id", "app_label", args.ts_col]]]

    print(f"[make_splits] {len(df)} flows  |  {df['app_label'].nunique()} classes")
    print(df["app_label"].value_counts().to_string())

    if args.strategy == "time":
        split_ids = time_split(df, args.ts_col)
    else:
        split_ids = random_split(df, seed=args.seed)

    if not args.no_fewshot:
        split_ids, fewshot_ids, held_out = carve_fewshot(df, split_ids, seed=args.seed)
        _write_ids(out_dir / "fewshot.txt", fewshot_ids)
        (out_dir / "fewshot_apps.txt").write_text("\n".join(held_out) + "\n")
        print(f"[make_splits] Few-shot held-out classes: {held_out}")

    _write_ids(out_dir / "train.txt", split_ids["train"])
    _write_ids(out_dir / "val.txt",   split_ids["val"])
    _write_ids(out_dir / "test.txt",  split_ids["test"])

    # update 'split' column in parquet
    id_to_split = {}
    for s, ids in split_ids.items():
        for fid in ids:
            id_to_split[fid] = s
    if not args.no_fewshot:
        for fid in fewshot_ids:
            id_to_split[fid] = "fewshot"

    df["split"] = df["flow_id"].map(id_to_split).fillna("")
    df.to_parquet(args.parquet, index=False)
    print(f"[make_splits] Updated 'split' column in {args.parquet}")


if __name__ == "__main__":
    main()
