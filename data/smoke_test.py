"""
Phase 0 smoke-test — run with:

    python -m data.smoke_test

Checks every component of the data pipeline end-to-end:
  1. FlowFeatureExtractor.extract()
  2. derive_context_from_packets()
  3. FlowDataset from raw dicts
  4. FlowDataset from CSV (round-trip export → reload)
  5. collate_flows padding + mask
  6. build_loaders split sizes
  7. Tensor shapes fed into FlowContextEncoder
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def run_smoke_test() -> None:
    print("=" * 60)
    print("Phase 0 smoke-test")
    print("=" * 60)

    # ── 1. Feature extractor ──────────────────────────────────────────
    from data.features import (
        FlowFeatureExtractor,
        derive_context_from_packets,
        PACKET_FEAT_DIM,
        CTX_FEAT_DIM,
    )

    extractor = FlowFeatureExtractor(max_packets=128, min_packets=4)
    dummy_pkts = [
        {"direction": 1,  "size": 60,   "timestamp": 0.000, "tcp_flags": 0x02},
        {"direction": -1, "size": 60,   "timestamp": 0.015, "tcp_flags": 0x12},
        {"direction": 1,  "size": 1460, "timestamp": 0.020, "tcp_flags": 0x10},
        {"direction": -1, "size": 200,  "timestamp": 0.040, "tcp_flags": 0x10},
    ]
    dummy_meta = {"rtt_ms": 15.0, "jitter_ms": 2.0, "pkt_loss_rate": 0.01, "throughput": 500_000}
    pkt_t, ctx_t = extractor.extract(dummy_pkts, dummy_meta)
    assert pkt_t.shape == (4, PACKET_FEAT_DIM), f"pkt shape wrong: {pkt_t.shape}"
    assert ctx_t.shape == (CTX_FEAT_DIM,),       f"ctx shape wrong: {ctx_t.shape}"
    print(f"[1] Feature extractor OK  pkt={pkt_t.shape}  ctx={ctx_t.shape}")

    derived = derive_context_from_packets(dummy_pkts)
    assert "rtt_ms" in derived and "throughput" in derived
    print(f"[2] derive_context OK  {derived}")

    # ── 3. FlowDataset from raw dicts ─────────────────────────────────
    from data.synthetic import make_synthetic_dataset, CLASS_NAMES
    from data.dataset   import FlowDataset, collate_flows

    records = make_synthetic_dataset(n_per_class=50, seed=0)
    ds = FlowDataset(records, max_packets=64)
    assert len(ds) > 0, "Dataset is empty"
    pkt_s, ctx_s, lbl_s = ds[0]
    assert pkt_s.ndim == 2 and pkt_s.shape[1] == PACKET_FEAT_DIM
    assert ctx_s.shape == (CTX_FEAT_DIM,)
    print(f"[3] FlowDataset (raw)  len={len(ds)}  sample pkt={pkt_s.shape}  label={lbl_s}")

    # ── 4. CSV round-trip ─────────────────────────────────────────────
    from data.io_csv import export_to_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "flows.csv"
        export_to_csv(records[:100], csv_path)
        ds_csv = FlowDataset(csv_path, max_packets=64)
        assert len(ds_csv) > 0
        print(f"[4] CSV round-trip OK  len={len(ds_csv)}")

    # ── 5. collate_flows ──────────────────────────────────────────────
    import torch

    batch = [ds[i] for i in range(8)]
    pkts_b, ctx_b, mask_b, labels_b = collate_flows(batch)
    assert pkts_b.ndim  == 3
    assert ctx_b.ndim   == 2
    assert mask_b.ndim  == 2
    assert mask_b.dtype == torch.bool
    assert labels_b.ndim == 1
    print(f"[5] collate_flows OK  pkts={pkts_b.shape}  mask={mask_b.shape}")

    # ── 6. build_loaders ──────────────────────────────────────────────
    from data.loaders import build_loaders

    train_dl, val_dl, test_dl = build_loaders(
        records, batch_size=32, max_packets=64, seed=42
    )
    train_batch = next(iter(train_dl))
    assert len(train_batch) == 4
    print(f"[6] build_loaders OK  train batches={len(train_dl)}  val={len(val_dl)}  test={len(test_dl)}")

    # ── 7. Forward pass through FlowContextEncoder ───────────────────
    try:
        from model.encoder import FlowContextEncoder

        pkts_b, ctx_b, mask_b, labels_b = train_batch
        enc = FlowContextEncoder(
            d_in     = PACKET_FEAT_DIM,
            d_ctx    = CTX_FEAT_DIM,
            d_model  = 32,
            d_embed  = 64,
            n_layers = 2,
        )
        enc.eval()
        with torch.no_grad():
            z = enc(pkts_b, ctx_b, mask_b)
        assert z.shape == (pkts_b.shape[0], 64)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        print(f"[7] Forward pass OK  z={z.shape}  norms~1")
    except ImportError:
        print("[7] model.encoder not importable — skipped (expected if run standalone)")

    print("=" * 60)
    print("All Phase 0 checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
