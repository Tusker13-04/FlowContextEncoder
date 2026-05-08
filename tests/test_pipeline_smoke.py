"""
tests/test_pipeline_smoke.py
============================
Smoke tests that run on CPU with no external data dependencies.
These are what CI runs on every push to main.
"""

import numpy as np
import pandas as pd
import pytest
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_dummy_flow(app_label="video_streaming", split="train"):
    return {
        "flow_id": "deadbeef",
        "app_label": app_label,
        "packets": np.random.rand(128, 5).astype(np.float32),
        "rtt_ms": 40.0,
        "jitter_ms": 5.0,
        "pkt_loss_rate": 0.01,
        "throughput_kbps": 5000.0,
        "split": split,
    }


# ── scripts/build_flows.py ────────────────────────────────────────────────────

def test_packets_to_array_shape():
    from scripts.build_flows import _packets_to_array, _PacketRecord
    records = [
        _PacketRecord(ts=float(i) * 0.01, size=1000, direction=i % 2, tcp_flags=24)
        for i in range(200)
    ]
    arr = _packets_to_array(records)
    assert arr.shape == (128, 5), f"Expected (128,5), got {arr.shape}"
    assert arr.dtype == np.float32
    assert arr[:, 0].max() <= 1.0  # size normalised


def test_flow_context_returns_tuple():
    from scripts.build_flows import _flow_context, _PacketRecord
    records = [
        _PacketRecord(ts=float(i) * 0.05, size=800, direction=0)
        for i in range(20)
    ]
    rtt, jitter, loss, tput = _flow_context(records)
    assert tput > 0


def test_flow_key_is_bidirectional():
    from scripts.build_flows import _flow_key
    k1 = _flow_key("1.1.1.1", "2.2.2.2", 1234, 443, "TCP")
    k2 = _flow_key("2.2.2.2", "1.1.1.1", 443, 1234, "TCP")
    assert k1 == k2, "Flow key must be the same regardless of direction"


# ── data/fiveg_dataset.py ─────────────────────────────────────────────────────

def test_synthesise_packets_shape():
    from data.fiveg_dataset import _synthesise_packets
    arr = _synthesise_packets("gaming", n_pkts=64, duration_s=2.0)
    assert arr.shape == (128, 5)
    assert arr.dtype == np.float32


def test_load_5g_csv_with_dummy(tmp_path):
    from data.fiveg_dataset import load_5g_csv
    csv_path = tmp_path / "dummy_5g.csv"
    df_raw = pd.DataFrame({
        "Label": ["video_streaming", "gaming", "voip", "unknown_app"],
        "duration": [10.0, 5.0, 3.0, 2.0],
        "total_bytes": [10_000, 5_000, 2_000, 1_000],
        "total_pkts": [100, 50, 30, 10],
    })
    df_raw.to_csv(csv_path, index=False)
    result = load_5g_csv(csv_path, split="train")
    # 3 recognised labels (unknown_app skipped)
    assert len(result) == 3
    assert set(result["app_label"].unique()).issubset({"video_streaming", "gaming", "voip"})
    assert result["packets"].iloc[0].shape == (128, 5)


# ── data/cesnet_dataset.py ────────────────────────────────────────────────────

def test_map_app_to_coarse():
    from data.cesnet_dataset import _map_app_to_coarse
    assert _map_app_to_coarse("youtube") == "video_streaming"
    assert _map_app_to_coarse("steam") == "gaming"
    assert _map_app_to_coarse("zoom") == "voip"
    assert _map_app_to_coarse("background_traffic") is None


def test_cesnet_row_to_flow():
    from data.cesnet_dataset import _cesnet_row_to_flow
    row = {
        "PPI_PKT_LEN": np.random.randint(64, 1500, 50).tolist(),
        "PPI_IPT": np.random.exponential(20, 50).tolist(),
        "PPI_DIR": [0, 1] * 25,
        "PPI_FLAGS": [24] * 50,
        "QUIC_RTT": 35.0,
        "BYTES": 50_000,
        "DURATION": 2.0,
        "FLOW_ID": "test-flow-001",
    }
    flow = _cesnet_row_to_flow(row, "video_streaming", "train")
    assert flow is not None
    assert flow["packets"].shape == (128, 5)
    assert flow["app_label"] == "video_streaming"
    assert flow["rtt_ms"] == pytest.approx(35.0)


# ── scripts/make_splits.py ────────────────────────────────────────────────────

def test_make_splits_random(tmp_path):
    from scripts.make_splits import main as splits_main
    import sys

    # Build dummy parquet
    flows_dir = tmp_path / "flows"
    flows_dir.mkdir()
    rows = []
    for i in range(100):
        rows.append({
            "flow_id": f"flow_{i:04d}",
            "app_label": ["video_streaming", "gaming", "voip", "web"][i % 4],
        })
    pd.DataFrame(rows).to_parquet(flows_dir / "dummy.parquet", index=False)

    out_dir = tmp_path / "splits"
    sys.argv = [
        "make_splits.py",
        "--flow-dir", str(flows_dir),
        "--out-dir", str(out_dir),
        "--strategy", "random",
        "--fewshot-apps",  # no fewshot apps for this test
    ]
    splits_main()

    train_ids = (out_dir / "train.txt").read_text().splitlines()
    val_ids = (out_dir / "val.txt").read_text().splitlines()
    test_ids = (out_dir / "test.txt").read_text().splitlines()

    all_ids = set(train_ids) | set(val_ids) | set(test_ids)
    assert len(all_ids) == 100, "All flow IDs must appear exactly once"
    assert len(set(train_ids) & set(val_ids)) == 0, "Train and val must not overlap"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val and test must not overlap"
