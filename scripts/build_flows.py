"""
build_flows.py  —  pcap / Zeek conn.log  →  per-split Parquet files

Usage
-----
  # From raw pcap
  python scripts/build_flows.py --input data/raw/capture.pcap --out data/flows/lab.parquet --label youtube

  # From Zeek conn.log
  python scripts/build_flows.py --input data/raw/conn.log --zeek --out data/flows/zeek.parquet

Output schema (one row per flow)
---------------------------------
  flow_id         : str
  app_label       : str
  packets         : np.ndarray  shape (N_MAX, 5)  —  [size, direction, iat, tcp_flags, quic_type]
  rtt_ms          : float
  jitter_ms       : float
  pkt_loss_rate   : float
  throughput_kbps : float
  split           : str   —  filled by make_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_MAX = 128           # truncate / pad every flow to this many packets
INACTIVITY_TIMEOUT = 30.0  # seconds  —  new flow if gap > this

# Feature indices in the per-packet array
IDX_SIZE      = 0
IDX_DIR       = 1   # +1 = client→server, -1 = server→client
IDX_IAT       = 2   # inter-arrival time in ms
IDX_FLAGS     = 3   # TCP flags byte (0–255); 0 for UDP/QUIC
IDX_QUIC_TYPE = 4   # QUIC packet type nibble; 0 for non-QUIC

# Coarse label mapping used by downstream loaders too
COARSE_MAP: Dict[str, str] = {
    # video streaming
    "youtube":           "video_streaming",
    "netflix":           "video_streaming",
    "twitch":            "video_streaming",
    "disneyplus":        "video_streaming",
    "primevideo":        "video_streaming",
    # gaming
    "gaming":            "gaming",
    "steam":             "gaming",
    "valorant":          "gaming",
    "csgo":              "gaming",
    # voip
    "voip":              "voip",
    "zoom":              "voip",
    "teams":             "voip",
    "discord":           "voip",
    # web
    "web":               "web",
    "http":              "web",
    "https":             "web",
    # xr
    "xr":                "xr",
    "vr":                "xr",
    "ar":                "xr",
}


# ──────────────────────────────────────────────────────────────────────────────
# Flow key helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flow_key(src_ip: str, dst_ip: str, sport: int, dport: int, proto: int) -> str:
    """Canonical bidirectional 5-tuple key."""
    a = (src_ip, sport)
    b = (dst_ip, dport)
    if a > b:
        a, b = b, a
    raw = f"{a[0]}:{a[1]}-{b[0]}:{b[1]}-{proto}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _flow_id(key: str, start_ts: float) -> str:
    return f"{key}_{int(start_ts * 1000)}"


# ──────────────────────────────────────────────────────────────────────────────
# Packet-level parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_pcap(path: str) -> List[dict]:
    """
    Parse a pcap file using dpkt (optional) or scapy fallback.
    Returns a list of packet dicts with keys:
        ts, src, dst, sport, dport, proto, size, tcp_flags, is_quic, quic_type, direction
    direction is determined later during flow assembly.
    """
    packets: List[dict] = []
    try:
        import dpkt  # type: ignore
        with open(path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                        continue
                    proto = ip.p if hasattr(ip, "p") else ip.nxt
                    src = str(ip.src) if isinstance(ip.src, str) else _ip_to_str(ip.src)
                    dst = str(ip.dst) if isinstance(ip.dst, str) else _ip_to_str(ip.dst)
                    tcp_flags = 0
                    quic_type = 0
                    is_quic = False
                    sport = dport = 0
                    if proto == 6:  # TCP
                        tcp = ip.data
                        sport, dport = tcp.sport, tcp.dport
                        tcp_flags = tcp.flags
                    elif proto == 17:  # UDP
                        udp = ip.data
                        sport, dport = udp.sport, udp.dport
                        if sport == 443 or dport == 443:
                            is_quic = True
                            if len(udp.data) > 0:
                                quic_type = (udp.data[0] >> 4) & 0xF
                    packets.append({
                        "ts": float(ts), "src": src, "dst": dst,
                        "sport": sport, "dport": dport, "proto": proto,
                        "size": len(buf), "tcp_flags": tcp_flags,
                        "is_quic": is_quic, "quic_type": quic_type,
                    })
                except Exception:
                    continue
    except ImportError:
        print("[build_flows] dpkt not found; falling back to scapy (slow).")
        from scapy.all import rdpcap, IP, TCP, UDP  # type: ignore
        raw = rdpcap(path)
        for pkt in raw:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            proto = ip.proto
            src, dst = ip.src, ip.dst
            sport = dport = 0
            tcp_flags = quic_type = 0
            is_quic = False
            if pkt.haslayer(TCP):
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
                tcp_flags = int(pkt[TCP].flags)
            elif pkt.haslayer(UDP):
                from scapy.all import UDP as SCAPY_UDP
                sport, dport = pkt[SCAPY_UDP].sport, pkt[SCAPY_UDP].dport
                if sport == 443 or dport == 443:
                    is_quic = True
            packets.append({
                "ts": float(pkt.time), "src": src, "dst": dst,
                "sport": sport, "dport": dport, "proto": proto,
                "size": len(bytes(pkt)), "tcp_flags": tcp_flags,
                "is_quic": is_quic, "quic_type": quic_type,
            })
    return sorted(packets, key=lambda x: x["ts"])


def _ip_to_str(raw: bytes) -> str:
    if len(raw) == 4:
        return ".".join(str(b) for b in raw)
    return ":".join(raw.hex()[i:i+4] for i in range(0, 32, 4))


def _parse_zeek_conn(path: str) -> List[dict]:
    """
    Parse a Zeek conn.log TSV.
    Returns flow-level records (each Zeek conn.log row = one flow).
    We synthesize a minimal packet-sequence from flow-level stats.
    """
    rows = []
    with open(path) as f:
        headers = None
        for line in f:
            line = line.rstrip()
            if line.startswith("#fields"):
                headers = line.split("\t")[1:]
            elif line.startswith("#"):
                continue
            elif headers:
                vals = line.split("\t")
                row = dict(zip(headers, vals))
                rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Flow assembly from packets
# ──────────────────────────────────────────────────────────────────────────────

class _FlowAccum:
    def __init__(self, key: str, first_pkt: dict):
        self.key = key
        self.start_ts = first_pkt["ts"]
        self.last_ts = first_pkt["ts"]
        self.client = (first_pkt["src"], first_pkt["sport"])
        self.pkts: List[Tuple[float, int, int, int, int, int]] = []  # (ts, size, dir, flags, quic_type, proto)
        self._add(first_pkt)

    def _add(self, p: dict):
        direction = 1 if (p["src"], p["sport"]) == self.client else -1
        self.pkts.append((p["ts"], p["size"], direction, p["tcp_flags"], p["quic_type"], p["proto"]))
        self.last_ts = p["ts"]

    def add(self, p: dict) -> bool:
        """Returns False if timeout exceeded (caller should start new flow)."""
        if p["ts"] - self.last_ts > INACTIVITY_TIMEOUT:
            return False
        self._add(p)
        return True

    def to_record(self, label: str) -> dict:
        n = len(self.pkts)
        if n == 0:
            return None
        ts_arr = np.array([p[0] for p in self.pkts], dtype=np.float64)
        sizes  = np.array([p[1] for p in self.pkts], dtype=np.float32)
        dirs   = np.array([p[2] for p in self.pkts], dtype=np.float32)
        flags  = np.array([p[3] for p in self.pkts], dtype=np.float32)
        qtypes = np.array([p[4] for p in self.pkts], dtype=np.float32)

        # inter-arrival times in ms (first IAT = 0)
        iats = np.zeros(n, dtype=np.float32)
        if n > 1:
            iats[1:] = np.diff(ts_arr) * 1000.0

        # build (N, 5) feature matrix
        feat = np.stack([sizes, dirs, iats, flags, qtypes], axis=1)  # (n, 5)

        # truncate or pad to N_MAX
        if n >= N_MAX:
            feat = feat[:N_MAX]
        else:
            pad = np.zeros((N_MAX - n, 5), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)

        # flow-level context features
        duration = max(ts_arr[-1] - ts_arr[0], 1e-6)
        total_bytes = float(sizes.sum())
        throughput_kbps = (total_bytes * 8) / (duration * 1000.0)

        # RTT estimate: use first SYN→SYN-ACK gap if TCP, else heuristic
        rtt_ms = float(np.median(iats[iats > 0])) if (iats > 0).any() else 20.0
        jitter_ms = float(np.std(iats[iats > 0])) if (iats > 0).sum() > 1 else 0.0

        # packet loss proxy: fraction of retransmission-hinting TCP flags (RST/FIN bursts)
        pkt_loss_rate = 0.0
        if n > 4:
            # rough: packets with size < 64 bytes after the first few
            tiny = (sizes[4:] < 64).sum()
            pkt_loss_rate = float(tiny) / max(n - 4, 1)

        return {
            "flow_id":         _flow_id(self.key, self.start_ts),
            "app_label":       label,
            "packets":         feat,
            "rtt_ms":          rtt_ms,
            "jitter_ms":       jitter_ms,
            "pkt_loss_rate":   pkt_loss_rate,
            "throughput_kbps": throughput_kbps,
            "split":           "",
        }


def build_flows_from_packets(packets: List[dict], label: str) -> pd.DataFrame:
    active: Dict[str, _FlowAccum] = {}
    finished: List[dict] = []

    for p in packets:
        key = _flow_key(p["src"], p["dst"], p["sport"], p["dport"], p["proto"])
        if key in active:
            ok = active[key].add(p)
            if not ok:
                rec = active[key].to_record(label)
                if rec:
                    finished.append(rec)
                active[key] = _FlowAccum(key, p)
        else:
            active[key] = _FlowAccum(key, p)

    for acc in active.values():
        rec = acc.to_record(label)
        if rec:
            finished.append(rec)

    if not finished:
        return pd.DataFrame()
    return pd.DataFrame(finished)


def build_flows_from_zeek(rows: List[dict], label: str) -> pd.DataFrame:
    """
    Zeek conn.log already has flow-level stats; synthesise a packet sequence
    from orig_pkts / resp_pkts / duration / orig_bytes / resp_bytes.
    """
    records = []
    for row in rows:
        try:
            ts         = float(row.get("ts", 0))
            duration   = max(float(row.get("duration", 1) or 1), 1e-3)
            orig_pkts  = int(row.get("orig_pkts", 1) or 1)
            resp_pkts  = int(row.get("resp_pkts", 0) or 0)
            orig_bytes = float(row.get("orig_bytes", 0) or 0)
            resp_bytes = float(row.get("resp_bytes", 0) or 0)
            proto_str  = row.get("proto", "tcp").lower()
            proto      = 6 if proto_str == "tcp" else 17
            src        = row.get("id.orig_h", "0.0.0.0")
            dst        = row.get("id.resp_h", "0.0.0.0")
            sport      = int(row.get("id.orig_p", 0) or 0)
            dport      = int(row.get("id.resp_p", 0) or 0)
        except (ValueError, TypeError):
            continue

        n_total = min(orig_pkts + resp_pkts, N_MAX)
        if n_total == 0:
            continue

        # Synthesise packet sizes from byte totals
        mean_orig = orig_bytes / max(orig_pkts, 1)
        mean_resp = resp_bytes / max(resp_pkts, 1)
        np.random.seed(int(ts * 1000) % (2**31))
        sizes_fwd = np.random.normal(mean_orig, mean_orig * 0.2, orig_pkts).clip(40, 1500)
        sizes_rev = np.random.normal(mean_resp, mean_resp * 0.2, resp_pkts).clip(40, 1500)
        all_sizes = np.concatenate([sizes_fwd, sizes_rev])[:N_MAX].astype(np.float32)
        dirs      = np.concatenate([np.ones(orig_pkts), -np.ones(resp_pkts)])[:N_MAX].astype(np.float32)
        mean_iat  = duration * 1000.0 / max(n_total - 1, 1)
        iats      = np.abs(np.random.normal(mean_iat, mean_iat * 0.3, N_MAX)).astype(np.float32)
        iats[0]   = 0.0
        flags     = np.zeros(N_MAX, dtype=np.float32)
        qtypes    = np.zeros(N_MAX, dtype=np.float32)

        n_pad = N_MAX - len(all_sizes)
        if n_pad > 0:
            all_sizes = np.concatenate([all_sizes, np.zeros(n_pad, np.float32)])
            dirs      = np.concatenate([dirs,      np.zeros(n_pad, np.float32)])

        feat = np.stack([all_sizes, dirs, iats, flags, qtypes], axis=1)

        total_bytes    = orig_bytes + resp_bytes
        throughput_kbps = (total_bytes * 8) / (duration * 1000.0)
        rtt_ms          = float(iats[iats > 0].mean()) if (iats > 0).any() else 20.0
        jitter_ms       = float(iats[iats > 0].std())  if (iats > 0).sum() > 1 else 0.0

        key = _flow_key(src, dst, sport, dport, proto)
        records.append({
            "flow_id":         _flow_id(key, ts),
            "app_label":       label,
            "packets":         feat,
            "rtt_ms":          rtt_ms,
            "jitter_ms":       jitter_ms,
            "pkt_loss_rate":   0.0,
            "throughput_kbps": throughput_kbps,
            "split":           "",
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build flow Parquet from pcap / Zeek conn.log")
    parser.add_argument("--input",  required=True, help="Path to pcap or Zeek conn.log")
    parser.add_argument("--out",    required=True, help="Output Parquet path")
    parser.add_argument("--label",  default="unknown", help="App label (e.g. youtube)")
    parser.add_argument("--zeek",   action="store_true", help="Input is a Zeek conn.log, not pcap")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_flows] Reading {args.input} ...")
    if args.zeek:
        rows = _parse_zeek_conn(args.input)
        label = COARSE_MAP.get(args.label.lower(), args.label)
        df = build_flows_from_zeek(rows, label)
    else:
        pkts = _parse_pcap(args.input)
        label = COARSE_MAP.get(args.label.lower(), args.label)
        df = build_flows_from_packets(pkts, label)

    if df.empty:
        print("[build_flows] WARNING: no flows extracted.")
        return

    print(f"[build_flows] {len(df)} flows  →  {args.out}")
    df.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
