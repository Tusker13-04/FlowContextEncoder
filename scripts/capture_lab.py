"""
scripts/capture_lab.py
======================
Lab packet-capture wrapper for manual ground-truth collection.

Workflow
--------
1. Optionally apply tc-netem impairment (high RTT, packet loss, jitter).
2. Start tcpdump on the target interface.
3. Wait for the user to run the target app (YouTube, Netflix, game, etc.)
   — or sleep for --duration seconds in automated mode.
4. Stop tcpdump, remove impairment.
5. Call build_flows.py on the resulting pcap.

Usage
-----
# Manual interactive capture (good/good network):
    sudo python scripts/capture_lab.py \
        --label video_streaming \
        --split train \
        --iface eth0 \
        --duration 120

# Automated bad-network scenario (200ms RTT, 5% loss):
    sudo python scripts/capture_lab.py \
        --label gaming \
        --split test \
        --iface eth0 \
        --duration 120 \
        --netem "delay 200ms 20ms loss 5%"

Requires: tcpdump (system), tc (iproute2, Linux only).
Python deps: none beyond stdlib.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COARSE_CLASSES = {"video_streaming", "gaming", "voip", "web", "xr"}
OUTPUT_DIR = Path("data/flows")
RAW_DIR = Path("data/raw")


def _run(cmd: str, check: bool = True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  [ERROR] {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def apply_netem(iface: str, netem: str):
    """Apply tc-netem impairment to the given interface."""
    print(f"[capture_lab] Applying netem impairment: {netem}")
    _run(f"tc qdisc add dev {iface} root netem {netem}")


def remove_netem(iface: str):
    """Remove tc-netem impairment (best-effort)."""
    _run(f"tc qdisc del dev {iface} root netem", check=False)


def capture(
    iface: str,
    pcap_path: Path,
    duration: int,
    interactive: bool,
):
    """Start tcpdump, wait, then stop."""
    cmd = f"tcpdump -i {iface} -w {pcap_path} -q"
    print(f"[capture_lab] Starting capture → {pcap_path}")
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if interactive:
        input("  [capture_lab] Press ENTER when done running the app...")
    else:
        print(f"  [capture_lab] Capturing for {duration}s ...")
        time.sleep(duration)

    proc.terminate()
    proc.wait()
    print(f"[capture_lab] Capture complete. pcap saved to {pcap_path}")


def main():
    parser = argparse.ArgumentParser(description="Lab capture wrapper for ground-truth collection.")
    parser.add_argument("--label", required=True, choices=sorted(COARSE_CLASSES))
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "fewshot"])
    parser.add_argument("--iface", default="eth0", help="Network interface to capture on")
    parser.add_argument("--duration", type=int, default=120, help="Capture duration in seconds (automated mode)")
    parser.add_argument("--interactive", action="store_true", help="Wait for ENTER instead of fixed duration")
    parser.add_argument("--netem", default=None,
                        help="tc-netem impairment string, e.g. 'delay 200ms 20ms loss 5%%'. Omit for good-network scenario.")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory for flow parquet files")
    parser.add_argument("--keep-pcap", action="store_true", help="Keep raw pcap after converting to parquet")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    scenario = "bad" if args.netem else "good"
    pcap_name = f"{args.label}_{args.split}_{scenario}_{int(time.time())}.pcap"
    pcap_path = RAW_DIR / pcap_name

    print(f"[capture_lab] Label={args.label}  Split={args.split}  Network={scenario}")

    if args.netem:
        apply_netem(args.iface, args.netem)

    try:
        capture(args.iface, pcap_path, args.duration, args.interactive)
    finally:
        if args.netem:
            remove_netem(args.iface)

    # Convert pcap → parquet via build_flows.py
    cmd = (
        f"python scripts/build_flows.py "
        f"--source pcap "
        f"--input {pcap_path} "
        f"--label {args.label} "
        f"--split {args.split} "
        f"--output {args.output}"
    )
    print("[capture_lab] Converting pcap → parquet ...")
    _run(cmd)

    if not args.keep_pcap:
        pcap_path.unlink(missing_ok=True)
        print(f"[capture_lab] Removed pcap ({pcap_path}).")

    print("[capture_lab] Done.")


if __name__ == "__main__":
    main()
