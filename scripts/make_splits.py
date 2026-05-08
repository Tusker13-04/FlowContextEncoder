"""
scripts/make_splits.py
======================
Build leakage-resistant train/val/test/fewshot split index files.

Two split strategies
--------------------
1. Time-based (default for CESNET data):
   - Train  : weeks 1-3
   - Val    : week 4
   - Test   : week 5  (different calendar week — no temporal leakage)
   Any flow whose first-packet timestamp falls in a given week is assigned
   to that split.

2. App-held-out (few-shot evaluation):
   - 3 apps are never seen during train/val; reserved in fewshot.txt.
   - Remaining apps follow the time-based split above.
   - App names to hold out are written to splits/fewshot_apps.txt.

Output
------
  splits/train.txt     — flow_id per line
  splits/val.txt
  splits/test.txt
  splits/fewshot.txt
  splits/fewshot_apps.txt  — app labels held out for few-shot eval

Usage
-----
    python scripts/make_splits.py \
        --flow-dir data/flows/ \
        --out-dir  splits/ \
        --strategy time  \
        --fewshot-apps xr voip
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42


def _load_all_flows(flow_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(flow_dir.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["flow_id", "app_label"])
        # Attempt to read a timestamp column if present
        try:
            df_ts = pd.read_parquet(p, columns=["flow_id", "ts"])
            df["ts"] = df_ts["ts"]
        except Exception:
            df["ts"] = None
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {flow_dir}")
    return pd.concat(parts, ignore_index=True)


def _time_based_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign split based on week-of-year from 'ts' column.
    Falls back to a random 70/15/15 split if timestamps are unavailable.
    """
    if df["ts"].isna().all():
        print("[make_splits] No timestamps found — falling back to random 70/15/15 split.")
        rng = random.Random(RANDOM_SEED)
        flow_ids = df["flow_id"].tolist()
        rng.shuffle(flow_ids)
        n = len(flow_ids)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        split_map = {}
        for i, fid in enumerate(flow_ids):
            if i < train_end:
                split_map[fid] = "train"
            elif i < val_end:
                split_map[fid] = "val"
            else:
                split_map[fid] = "test"
        df["split"] = df["flow_id"].map(split_map)
        return df

    df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    df["week"] = df["ts"].dt.isocalendar().week
    min_week = df["week"].min()

    def _assign(week):
        rel = week - min_week
        if rel <= 2:   return "train"   # weeks 1-3
        elif rel == 3: return "val"     # week 4
        else:          return "test"    # week 5+

    df["split"] = df["week"].apply(_assign)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build leakage-resistant flow split index.")
    parser.add_argument("--flow-dir", default="data/flows", help="Directory containing flow parquet files")
    parser.add_argument("--out-dir", default="splits", help="Output directory for split txt files")
    parser.add_argument("--strategy", default="time", choices=["time", "random"],
                        help="time = week-based split; random = 70/15/15 random split")
    parser.add_argument("--fewshot-apps", nargs="*", default=["xr"],
                        help="App labels to hold out entirely for few-shot eval (default: xr)")
    args = parser.parse_args()

    flow_dir = Path(args.flow_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[make_splits] Loading flows from {flow_dir} ...")
    df = _load_all_flows(flow_dir)
    print(f"[make_splits] Loaded {len(df):,} flows across {df['app_label'].nunique()} app labels.")

    # ── Few-shot holdout ─────────────────────────────────────────────────────
    fewshot_apps = set(args.fewshot_apps)
    fewshot_mask = df["app_label"].isin(fewshot_apps)
    df_fewshot = df[fewshot_mask].copy()
    df_main = df[~fewshot_mask].copy()

    print(f"[make_splits] Holding out {len(df_fewshot):,} flows from apps: {fewshot_apps}")

    # ── Main split ───────────────────────────────────────────────────────────
    if args.strategy == "time":
        df_main = _time_based_split(df_main)
    else:
        # Override to random
        df_main["ts"] = None
        df_main = _time_based_split(df_main)

    # ── Write split files (flow IDs only) ────────────────────────────────────
    for split_name in ["train", "val", "test"]:
        ids = df_main[df_main["split"] == split_name]["flow_id"].tolist()
        out_path = out_dir / f"{split_name}.txt"
        out_path.write_text("\n".join(ids))
        print(f"[make_splits] {split_name:6s}: {len(ids):>8,} flows → {out_path}")

    fewshot_ids = df_fewshot["flow_id"].tolist()
    (out_dir / "fewshot.txt").write_text("\n".join(fewshot_ids))
    (out_dir / "fewshot_apps.txt").write_text("\n".join(sorted(fewshot_apps)))
    print(f"[make_splits] fewshot: {len(fewshot_ids):>8,} flows → {out_dir / 'fewshot.txt'}")
    print(f"[make_splits] Done.")


if __name__ == "__main__":
    main()
