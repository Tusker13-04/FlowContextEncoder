"""
scripts/build_flows.py
======================
Convert raw packet sources (pcap files or Zeek conn.log) into
standardised per-flow parquet files consumed by the training pipeline.

Usage
-----
# From pcap:
    python scripts/build_flows.py \
        --source pcap \
        --input  data/raw/youtube_hd.pcap \
        --label  video_streaming \
        --split  train \
        --output data/flows/

# From Zeek conn.log directory:
    python scripts/build_flows.py \
        --source zeek \
        --input  data/raw/zeek_logs/ \
        --label  gaming \
        --split  val \
        --output data/flows/

Output schema (parquet columns)
-------------------------------
  flow_id          str       unique <src>-<dst>-<sport>-<dport>-<proto>-<ts>
  app_label        str       coarse class label
  packets          object    numpy array shape (N=128, 5):
                               col 0 : packet size (bytes, normalised 0-1)
                               col 1 : direction   (0=client→server, 1=server→client)
                               col 2 : inter-arrival time (ms, log1p normalised)
                               col 3 : tcp_flags   (6-bit int; 0 if not TCP)
                               col 4 : quic_type   (0-8 enum; 0 if not QUIC)
  rtt_ms           float32   estimated RTT (ms); NaN if unavailable
  jitter_ms        float32   std-dev of per-packet RTT samples (ms)
  pkt_loss_rate    float32   fraction of retransmitted / lost packets [0,1]
  throughput_kbps  float32   bytes / duration * 8 / 1000
  split            str       train | val | test | fewshot
"""

import argparse
import hashlib
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Optional heavy imports (only needed at runtime) ──────────────────────────
try:
    from scapy.all import PcapReader, TCP, UDP, IP, IPv6
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    warnings.warn("scapy not installed — pcap source unavailable. Run: pip install scapy")

# ── Constants ────────────────────────────────────────────────────────────────
SEQ_LEN = 128           # Truncate/pad every flow to this many packets
INACTIVITY_TIMEOUT = 30 # seconds — flow ends after 30s of no traffic
MIN_PKTS = 4            # Discard flows shorter than this

COARSE_CLASSES = {"video_streaming", "gaming", "voip", "web", "xr"}

# TCP flag bit positions (in the 6-bit flags field)
_TCP_FLAGS = {"FIN": 0, "SYN": 1, "RST": 2, "PSH": 3, "ACK": 4, "URG": 5}


def _tcp_flags_int(flags_str: str) -> int:
    """Convert Scapy TCP flags string like 'SA' to 6-bit integer."""
    val = 0
    for ch in flags_str.upper():
        if ch == 'F': val |= 1 << 0
        elif ch == 'S': val |= 1 << 1
        elif ch == 'R': val |= 1 << 2
        elif ch == 'P': val |= 1 << 3
        elif ch == 'A': val |= 1 << 4
        elif ch == 'U': val |= 1 << 5
    return val


def _flow_key(src_ip: str, dst_ip: str, sport: int, dport: int, proto: str) -> Tuple:
    """Canonical bi-directional flow key (always smaller IP first)."""
    a = (src_ip, sport)
    b = (dst_ip, dport)
    if a > b:
        a, b = b, a
    return (*a, *b, proto)


def _flow_id(key: Tuple, ts: float) -> str:
    raw = "-".join(str(x) for x in key) + f"-{ts:.3f}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Pcap parsing ─────────────────────────────────────────────────────────────

class _PacketRecord:
    __slots__ = ("ts", "size", "direction", "tcp_flags", "is_quic")

    def __init__(self, ts, size, direction, tcp_flags=0, is_quic=False):
        self.ts = ts
        self.size = size
        self.direction = direction
        self.tcp_flags = tcp_flags
        self.is_quic = is_quic


def _extract_flows_from_pcap(pcap_path: str) -> Dict[Tuple, List[_PacketRecord]]:
    """Parse pcap and group packets into bi-directional flows."""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy is required for pcap parsing.")

    flows: Dict[Tuple, List[_PacketRecord]] = {}
    last_seen: Dict[Tuple, float] = {}

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
                continue

            ts = float(pkt.time)
            ip_layer = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            size = len(pkt)

            if pkt.haslayer(TCP):
                proto = "TCP"
                l4 = pkt.getlayer(TCP)
                sport, dport = l4.sport, l4.dport
                flags = _tcp_flags_int(str(l4.flags))
                is_quic = False
            elif pkt.haslayer(UDP):
                proto = "UDP"
                l4 = pkt.getlayer(UDP)
                sport, dport = l4.sport, l4.dport
                flags = 0
                is_quic = (dport == 443 or sport == 443)  # heuristic
            else:
                continue

            key = _flow_key(src_ip, dst_ip, sport, dport, proto)
            direction = 0 if (src_ip, sport) <= (dst_ip, dport) else 1

            # Inactivity timeout — start new flow
            if key in last_seen and (ts - last_seen[key]) > INACTIVITY_TIMEOUT:
                flows.pop(key, None)

            if key not in flows:
                flows[key] = []

            flows[key].append(_PacketRecord(
                ts=ts, size=size, direction=direction,
                tcp_flags=flags, is_quic=is_quic
            ))
            last_seen[key] = ts

    return flows


# ── Feature extraction ────────────────────────────────────────────────────────

def _packets_to_array(records: List[_PacketRecord]) -> np.ndarray:
    """
    Convert a list of _PacketRecord to a (SEQ_LEN, 5) float32 array.
    Truncates to SEQ_LEN or zero-pads if shorter.
    """
    n = len(records)
    arr = np.zeros((SEQ_LEN, 5), dtype=np.float32)

    # Normalisation constants
    MAX_SIZE = 1500.0  # MTU
    prev_ts = records[0].ts

    for i, rec in enumerate(records[:SEQ_LEN]):
        iat = rec.ts - prev_ts
        prev_ts = rec.ts
        arr[i, 0] = min(rec.size / MAX_SIZE, 1.0)           # size, normalised
        arr[i, 1] = float(rec.direction)                    # direction
        arr[i, 2] = float(np.log1p(iat * 1000))            # IAT in ms, log1p
        arr[i, 3] = float(rec.tcp_flags) / 63.0            # tcp flags, normalised
        arr[i, 4] = 1.0 if rec.is_quic else 0.0            # quic indicator

    return arr


def _flow_context(records: List[_PacketRecord]) -> Tuple[float, float, float, float]:
    """Compute (rtt_ms, jitter_ms, pkt_loss_rate, throughput_kbps) from records."""
    if len(records) < 2:
        return float("nan"), float("nan"), 0.0, 0.0

    iats = []
    prev_ts = records[0].ts
    for rec in records[1:]:
        iats.append((rec.ts - prev_ts) * 1000)  # ms
        prev_ts = rec.ts

    iats_arr = np.array(iats)
    # Estimate RTT as 2× median IAT (very rough; use SYN-ACK delta when available)
    rtt_ms = float(2.0 * np.median(iats_arr))
    jitter_ms = float(np.std(iats_arr))

    # Loss: fraction of retransmissions (TCP RST / FIN bursts as proxy)
    rst_count = sum(1 for r in records if r.tcp_flags & (1 << 2))  # RST flag
    pkt_loss_rate = float(rst_count / len(records))

    duration = records[-1].ts - records[0].ts
    total_bytes = sum(r.size for r in records)
    throughput_kbps = float((total_bytes * 8 / 1000) / max(duration, 1e-6))

    return rtt_ms, jitter_ms, pkt_loss_rate, throughput_kbps


# ── Builders ──────────────────────────────────────────────────────────────────

def flows_from_pcap(
    pcap_path: str,
    label: str,
    split: str,
) -> pd.DataFrame:
    """Full pipeline: pcap → DataFrame of flow rows."""
    raw_flows = _extract_flows_from_pcap(pcap_path)
    rows = []
    first_ts = min(recs[0].ts for recs in raw_flows.values() if recs)

    for key, records in raw_flows.items():
        if len(records) < MIN_PKTS:
            continue
        fid = _flow_id(key, records[0].ts)
        pkts = _packets_to_array(records)
        rtt, jitter, loss, tput = _flow_context(records)
        rows.append({
            "flow_id": fid,
            "app_label": label,
            "packets": pkts,
            "rtt_ms": rtt,
            "jitter_ms": jitter,
            "pkt_loss_rate": loss,
            "throughput_kbps": tput,
            "split": split,
        })

    return pd.DataFrame(rows)


def flows_from_zeek_conn(
    conn_log_path: str,
    label: str,
    split: str,
) -> pd.DataFrame:
    """
    Parse Zeek conn.log TSV/JSON into flow rows.
    Zeek already aggregates packets into flows, so we synthesise a
    packet sequence from summary statistics (size histogram).
    """
    df = pd.read_csv(
        conn_log_path,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "ts", "uid", "src_ip", "src_port", "dst_ip", "dst_port",
            "proto", "service", "duration", "orig_bytes", "resp_bytes",
            "conn_state", "missed_bytes",
        ],
        on_bad_lines="skip",
    )

    rows = []
    for _, row in df.iterrows():
        n_pkts = SEQ_LEN  # We synthesise SEQ_LEN pseudo-packets from flow stats
        pkts = np.zeros((SEQ_LEN, 5), dtype=np.float32)

        orig_bytes = float(row.get("orig_bytes") or 0)
        resp_bytes = float(row.get("resp_bytes") or 0)
        duration = float(row.get("duration") or 1)
        avg_size = (orig_bytes + resp_bytes) / max(n_pkts, 1) / 1500
        avg_iat = np.log1p((duration * 1000) / max(n_pkts - 1, 1))

        for i in range(SEQ_LEN):
            pkts[i, 0] = np.clip(avg_size + np.random.normal(0, 0.05), 0, 1)
            pkts[i, 1] = 0.0 if i % 3 != 2 else 1.0   # simple directionality heuristic
            pkts[i, 2] = max(avg_iat + np.random.normal(0, 0.1), 0)

        rtt_ms = float(duration * 500)  # rough RTT estimate
        jitter_ms = rtt_ms * 0.1
        loss = 0.0 if str(row.get("conn_state")) in {"SF", "S1"} else 0.05
        tput = (orig_bytes + resp_bytes) * 8 / 1000 / max(duration, 1e-6)

        rows.append({
            "flow_id": str(row.get("uid", "")) or _flow_id((row.get("src_ip"), row.get("src_port"), row.get("dst_ip"), row.get("dst_port"), row.get("proto")), float(row.get("ts", 0))),
            "app_label": label,
            "packets": pkts,
            "rtt_ms": rtt_ms,
            "jitter_ms": jitter_ms,
            "pkt_loss_rate": loss,
            "throughput_kbps": float(tput),
            "split": split,
        })

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build flow parquet from pcap or Zeek logs.")
    parser.add_argument("--source", choices=["pcap", "zeek"], required=True)
    parser.add_argument("--input", required=True, help="Path to pcap file or Zeek conn.log")
    parser.add_argument("--label", required=True, choices=sorted(COARSE_CLASSES))
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "fewshot"])
    parser.add_argument("--output", default="data/flows", help="Output directory for parquet files")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_flows] source={args.source}  input={args.input}  label={args.label}")

    if args.source == "pcap":
        df = flows_from_pcap(args.input, args.label, args.split)
    else:
        df = flows_from_zeek_conn(args.input, args.label, args.split)

    if df.empty:
        print("[build_flows] WARNING: no flows extracted — check input file.")
        return

    fname = f"{args.label}_{args.split}_{Path(args.input).stem}.parquet"
    out_path = out_dir / fname
    df.to_parquet(out_path, index=False)
    print(f"[build_flows] Wrote {len(df)} flows → {out_path}")


if __name__ == "__main__":
    main()
