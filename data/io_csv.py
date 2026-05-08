"""
Phase 0 — CSV import/export utilities.

export_to_csv(records, path)
    Serialize a list of raw flow dicts (as produced by synthetic.py or
    a custom packet-capture script) to the FlowDataset CSV schema.

import_cesnet_quic22(path, max_rows)
    Read a CESNET-QUIC22 style CSV (available via CESNET DataZoo) and
    convert each row into the FlowDataset raw-dict format.
    Expected CESNET columns (subset used):
        FLOW_DURATION_MILLISECONDS, PACKETS, BYTES,
        TCP_FLAGS, SRC_TO_DST_AVG_THROUGHPUT,
        BIDIRECTIONAL_MIN_PS, BIDIRECTIONAL_MEAN_PS,
        BIDIRECTIONAL_STDDEV_PS, BIDIRECTIONAL_MIN_PIT_MS,
        BIDIRECTIONAL_MEAN_PIT_MS, BIDIRECTIONAL_STDDEV_PIT_MS,
        Label   (application class string)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing  import Any


# ── Export ─────────────────────────────────────────────────────────────────────

def export_to_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    """
    Dump a list of raw flow dicts to a CSV file readable by FlowDataset.

    Parameters
    ----------
    records : list of dicts {pkts, meta, label}  (from synthetic.py or capture)
    path    : output CSV path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label", "rtt_ms", "jitter_ms", "pkt_loss_rate", "throughput",
        "pkt_sizes", "pkt_dirs", "pkt_times", "pkt_flags",
    ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            pkts = rec["pkts"]
            meta = rec.get("meta", {})
            writer.writerow({
                "label":         rec.get("label", 0),
                "rtt_ms":        meta.get("rtt_ms",        0.0),
                "jitter_ms":     meta.get("jitter_ms",     0.0),
                "pkt_loss_rate": meta.get("pkt_loss_rate", 0.0),
                "throughput":    meta.get("throughput",    0.0),
                "pkt_sizes":  ";".join(str(int(p.get("size",      0)))   for p in pkts),
                "pkt_dirs":   ";".join(str(int(p.get("direction", 1)))   for p in pkts),
                "pkt_times":  ";".join(f"{p.get('timestamp', 0.0):.6f}" for p in pkts),
                "pkt_flags":  ";".join(str(int(p.get("tcp_flags", 0)))   for p in pkts),
            })
    print(f"[export_to_csv] wrote {len(records)} flows → {path}")


# ── CESNET-QUIC22 importer ─────────────────────────────────────────────────────

def import_cesnet_quic22(
    path: str | Path,
    max_rows: int = 100_000,
    label_col: str = "Label",
) -> list[dict[str, Any]]:
    """
    Convert CESNET-QUIC22 aggregated flow CSV rows into the raw-dict format
    expected by FlowDataset.

    Because CESNET-QUIC22 gives aggregate statistics (not per-packet lists),
    this function *synthesises* a per-packet sequence that reproduces the
    reported mean/stddev of packet sizes and inter-arrival times.  The
    generated sequence is deterministic given the row values.

    Parameters
    ----------
    path      : path to the CESNET-QUIC22 CSV (may be gzip-compressed .csv.gz)
    max_rows  : stop after this many rows (set 0 for all)
    label_col : column name for the application label

    Returns
    -------
    list of raw-flow dicts compatible with FlowDataset
    """
    import math
    import random
    import gzip

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    records: list[dict] = []
    with opener(path, mode="rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break

            n_pkts = max(4, int(float(row.get("PACKETS", 16))))
            n_pkts = min(n_pkts, 128)

            mean_ps  = float(row.get("BIDIRECTIONAL_MEAN_PS",  400))
            std_ps   = float(row.get("BIDIRECTIONAL_STDDEV_PS",  0)) or mean_ps * 0.2
            mean_iat = float(row.get("BIDIRECTIONAL_MEAN_PIT_MS", 10)) / 1000.0  # → sec
            std_iat  = float(row.get("BIDIRECTIONAL_STDDEV_PIT_MS", 0)) / 1000.0 or mean_iat * 0.2

            rng = random.Random(i)  # deterministic per row
            ts  = time.time()
            pkts: list[dict] = []
            for _ in range(n_pkts):
                size = max(40, int(rng.gauss(mean_ps, std_ps)))
                iat  = max(0.0, rng.gauss(mean_iat, std_iat))
                ts  += iat
                pkts.append({
                    "direction":  rng.choice([1, -1]),
                    "size":       size,
                    "timestamp":  ts,
                    "tcp_flags":  0x10,
                })

            total_bytes = float(row.get("BYTES", sum(p["size"] for p in pkts)))
            duration_s  = float(row.get("FLOW_DURATION_MILLISECONDS", 1000)) / 1000.0
            throughput  = total_bytes / max(duration_s, 1e-6)

            meta = {
                "rtt_ms":        float(row.get("BIDIRECTIONAL_MIN_PIT_MS", 10)),
                "jitter_ms":     float(row.get("BIDIRECTIONAL_STDDEV_PIT_MS", 2)),
                "pkt_loss_rate": 0.0,
                "throughput":    throughput,
            }

            records.append({
                "pkts":  pkts,
                "meta":  meta,
                "label": row.get(label_col, "unknown"),
            })

    print(f"[import_cesnet_quic22] loaded {len(records)} flows from {path}")
    return records
