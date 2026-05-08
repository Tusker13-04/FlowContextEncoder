"""
data/fiveg_dataset.py
=====================
Loader for the Kaggle 5G Traffic Dataset
(https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets).

After downloading, the dataset contains CSV files with per-flow features
such as packet counts, byte counts, duration, and protocol info.
This loader:
  1. Reads the CSVs.
  2. Maps known app columns to coarse classes.
  3. Synthesises per-packet sequences (realistic IAT/size distributions
     per class) when raw per-packet data is absent.
  4. Adds synthetic RTT/jitter sampled from gamma distributions with
     class-realistic parameters.
  5. Saves to data/flows/*.parquet in the standard schema.

Usage
-----
    python data/fiveg_dataset.py \
        --csv-dir  data/raw/5g/ \
        --output   data/flows/ \
        --split    train
"""

import argparse
import hashlib
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SEQ_LEN = 128
COARSE_CLASSES = ["video_streaming", "gaming", "voip", "web", "xr"]

# ── Coarse class name mapping ─────────────────────────────────────────────────
_5G_LABEL_MAP = {
    # Video streaming
    "video": "video_streaming",
    "streaming": "video_streaming",
    "youtube": "video_streaming",
    "netflix": "video_streaming",
    "twitch": "video_streaming",
    # Gaming
    "game": "gaming",
    "gaming": "gaming",
    # VoIP
    "voip": "voip",
    "call": "voip",
    "voice": "voip",
    # Web / general
    "web": "web",
    "http": "web",
    "browsing": "web",
    # XR
    "xr": "xr",
    "vr": "xr",
    "ar": "xr",
}

# ── Per-class realistic RTT/jitter parameters (gamma distribution) ────────────
# shape, scale → mean = shape * scale
_CLASS_RTT_PARAMS = {
    "video_streaming": (4.0, 10.0),   # mean ~40ms
    "gaming":          (2.0, 8.0),    # mean ~16ms — gamers demand low RTT
    "voip":            (3.0, 6.0),    # mean ~18ms
    "web":             (5.0, 12.0),   # mean ~60ms
    "xr":              (2.5, 5.0),    # mean ~12ms — XR is latency-critical
}
_CLASS_JITTER_PARAMS = {
    "video_streaming": (2.0, 3.0),    # mean ~6ms jitter
    "gaming":          (1.5, 1.5),    # mean ~2ms jitter
    "voip":            (1.5, 2.0),    # mean ~3ms jitter
    "web":             (3.0, 5.0),    # mean ~15ms jitter
    "xr":              (1.5, 1.0),    # mean ~1.5ms jitter
}

# ── Per-class packet-size distributions (normal, clipped to [64, 1500]) ───────
_CLASS_PKT_SIZE_MEAN = {
    "video_streaming": 1200,
    "gaming":           400,
    "voip":             200,
    "web":              800,
    "xr":               600,
}
_CLASS_PKT_SIZE_STD = {
    "video_streaming": 200,
    "gaming":          150,
    "voip":             80,
    "web":             300,
    "xr":              200,
}


def _map_label(raw_label: str) -> Optional[str]:
    lbl = str(raw_label).lower().strip()
    for key, coarse in _5G_LABEL_MAP.items():
        if key in lbl:
            return coarse
    return None


def _synthesise_packets(coarse: str, n_pkts: int, duration_s: float) -> np.ndarray:
    """
    Synthesise a (SEQ_LEN, 5) packet array from flow-level statistics.
    Packet sizes and IATs are drawn from class-realistic distributions.
    """
    arr = np.zeros((SEQ_LEN, 5), dtype=np.float32)
    n = min(n_pkts, SEQ_LEN)

    # Packet sizes
    mu = _CLASS_PKT_SIZE_MEAN.get(coarse, 800)
    sigma = _CLASS_PKT_SIZE_STD.get(coarse, 200)
    sizes = np.clip(np.random.normal(mu, sigma, n), 64, 1500)
    arr[:n, 0] = (sizes / 1500.0).astype(np.float32)

    # Direction: rough heuristic — server sends larger packets
    arr[:n, 1] = (sizes < mu).astype(np.float32)  # 1=client→server (smaller pkts)

    # IAT
    avg_iat_ms = (duration_s * 1000) / max(n - 1, 1)
    iats = np.clip(np.random.exponential(avg_iat_ms, n), 0, None)
    arr[:n, 2] = np.log1p(iats).astype(np.float32)

    # TCP flags (simplified: most packets are PSH+ACK = 0b011000 = 24)
    arr[:n, 3] = 24.0 / 63.0

    # QUIC indicator: 5G datasets are often QUIC-heavy
    arr[:n, 4] = 1.0 if coarse in {"video_streaming", "web"} else 0.0

    return arr


def _synthesise_context(coarse: str) -> dict:
    """Sample RTT, jitter, loss, throughput from class-realistic distributions."""
    rtt_shape, rtt_scale = _CLASS_RTT_PARAMS.get(coarse, (4.0, 10.0))
    jitter_shape, jitter_scale = _CLASS_JITTER_PARAMS.get(coarse, (2.0, 3.0))
    return {
        "rtt_ms": float(np.random.gamma(rtt_shape, rtt_scale)),
        "jitter_ms": float(np.random.gamma(jitter_shape, jitter_scale)),
        "pkt_loss_rate": float(np.clip(np.random.beta(1, 50), 0, 0.2)),  # ~2% mean
        # throughput will be computed from actual flow bytes/duration below
    }


def load_5g_csv(
    csv_path: Path,
    split: str,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Load one 5G Traffic Dataset CSV and return a DataFrame in standard flow schema.
    """
    np.random.seed(rng_seed)

    try:
        raw = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception as e:
        warnings.warn(f"Could not read {csv_path}: {e}")
        return pd.DataFrame()

    # ── Detect label column ───────────────────────────────────────────────────
    label_col = None
    for candidate in ["Label", "label", "class", "Class", "Category", "category", "app", "App"]:
        if candidate in raw.columns:
            label_col = candidate
            break
    if label_col is None:
        warnings.warn(f"No label column found in {csv_path}. Skipping.")
        return pd.DataFrame()

    # ── Detect numeric feature columns ───────────────────────────────────────
    duration_col = next((c for c in raw.columns if "duration" in c.lower()), None)
    bytes_col = next((c for c in raw.columns if "byte" in c.lower() or "len" in c.lower()), None)
    pkt_col = next((c for c in raw.columns if "pkt" in c.lower() or "packet" in c.lower()), None)

    rows = []
    for _, row in raw.iterrows():
        coarse = _map_label(str(row[label_col]))
        if coarse is None:
            continue

        n_pkts = int(row[pkt_col]) if pkt_col and not pd.isna(row.get(pkt_col)) else SEQ_LEN
        duration = float(row[duration_col]) if duration_col and not pd.isna(row.get(duration_col)) else 1.0
        total_bytes = float(row[bytes_col]) if bytes_col and not pd.isna(row.get(bytes_col)) else 1000.0

        pkts = _synthesise_packets(coarse, n_pkts, duration)
        ctx = _synthesise_context(coarse)
        ctx["throughput_kbps"] = (total_bytes * 8 / 1000) / max(duration, 1e-6)

        fid_raw = f"{csv_path.stem}-{coarse}-{len(rows)}"
        fid = hashlib.md5(fid_raw.encode()).hexdigest()[:16]

        rows.append({
            "flow_id": fid,
            "app_label": coarse,
            "packets": pkts,
            "rtt_ms": ctx["rtt_ms"],
            "jitter_ms": ctx["jitter_ms"],
            "pkt_loss_rate": ctx["pkt_loss_rate"],
            "throughput_kbps": ctx["throughput_kbps"],
            "split": split,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Load 5G Traffic Dataset CSVs → parquet flows.")
    parser.add_argument("--csv-dir", default="data/raw/5g", help="Directory containing 5G CSV files")
    parser.add_argument("--output", default="data/flows", help="Output directory for parquet files")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "fewshot"])
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"[fiveg_dataset] No CSV files found in {csv_dir}. Download from Kaggle first.")
        print("  https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets")
        return

    total_flows = 0
    for csv_path in sorted(csv_files):
        print(f"[fiveg_dataset] Processing {csv_path.name} ...")
        df = load_5g_csv(csv_path, args.split)
        if df.empty:
            continue
        out_path = out_dir / f"5g_{csv_path.stem}_{args.split}.parquet"
        df.to_parquet(out_path, index=False)
        total_flows += len(df)
        print(f"  → {len(df)} flows saved to {out_path}")

    print(f"[fiveg_dataset] Done. Total flows: {total_flows:,}")


if __name__ == "__main__":
    main()
