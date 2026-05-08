"""
capture_lab.py  —  tcpdump wrapper for labelled manual captures

Usage
-----
  # Good network
  python scripts/capture_lab.py --app youtube --duration 120 --out data/raw/youtube_good.pcap

  # Bad network (tc-netem impairment)
  python scripts/capture_lab.py --app netflix --duration 120 \
      --out data/raw/netflix_bad.pcap \
      --netem "delay 80ms 20ms distribution normal loss 2%"

Requires:
  - tcpdump installed and in PATH (sudo if capturing on a live interface)
  - tc (iproute2) for netem impairment (Linux only)
  - python scripts/build_flows.py available
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


NETEM_IFACE = os.environ.get("NETEM_IFACE", "eth0")   # interface to impair
DEFAULT_IFACE = os.environ.get("CAPTURE_IFACE", "any") # interface to capture


def _run(cmd: list[str], check: bool = True, sudo: bool = False) -> subprocess.CompletedProcess:
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    print(f"[capture_lab] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def apply_netem(netem_spec: str):
    """Apply tc-netem impairment. Idempotent — clears existing qdiscs first."""
    _run(["tc", "qdisc", "del", "dev", NETEM_IFACE, "root"], check=False, sudo=True)
    _run(
        ["tc", "qdisc", "add", "dev", NETEM_IFACE, "root", "netem"] + netem_spec.split(),
        sudo=True,
    )
    print(f"[capture_lab] Netem applied on {NETEM_IFACE}: {netem_spec}")


def remove_netem():
    """Remove all netem qdiscs from the interface."""
    _run(["tc", "qdisc", "del", "dev", NETEM_IFACE, "root"], check=False, sudo=True)
    print(f"[capture_lab] Netem removed from {NETEM_IFACE}")


def start_capture(out_pcap: str, iface: str) -> subprocess.Popen:
    if not shutil.which("tcpdump"):
        raise RuntimeError("tcpdump not found in PATH")
    cmd = ["tcpdump", "-i", iface, "-w", out_pcap, "-n", "-q"]
    if os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    print(f"[capture_lab] Starting capture on {iface} → {out_pcap}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_capture(proc: subprocess.Popen):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("[capture_lab] Capture stopped.")


def main():
    parser = argparse.ArgumentParser(description="Labelled pcap capture with optional netem impairment")
    parser.add_argument("--app",      required=True, help="App label (e.g. youtube, netflix, gaming)")
    parser.add_argument("--duration", type=int, default=120, help="Capture duration in seconds")
    parser.add_argument("--out",      required=True, help="Output pcap path")
    parser.add_argument("--iface",    default=DEFAULT_IFACE, help="Capture interface")
    parser.add_argument("--netem",    default=None,
                        help="tc-netem spec, e.g. 'delay 80ms 20ms loss 2%%'. Omit for good-network.")
    parser.add_argument("--flows-out", default=None,
                        help="If set, automatically run build_flows.py and write Parquet here.")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip automatic flow building after capture.")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scenario = "bad_network" if args.netem else "good_network"
    print(f"[capture_lab] App={args.app}  Scenario={scenario}  Duration={args.duration}s")

    if args.netem:
        apply_netem(args.netem)

    proc = start_capture(args.out, args.iface)
    print(f"[capture_lab] >>> Open {args.app} NOW — capturing for {args.duration}s ...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("[capture_lab] Interrupted by user.")
    finally:
        stop_capture(proc)
        if args.netem:
            remove_netem()

    print(f"[capture_lab] Pcap saved: {args.out}")

    # Optionally run flow builder
    if not args.no_build:
        flows_out = args.flows_out or args.out.replace(".pcap", "_flows.parquet")
        build_cmd = [
            sys.executable, "scripts/build_flows.py",
            "--input", args.out,
            "--out",   flows_out,
            "--label", args.app,
        ]
        print(f"[capture_lab] Building flows → {flows_out}")
        subprocess.run(build_cmd, check=False)

    # Emit metadata sidecar
    meta_path = args.out.replace(".pcap", "_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"app={args.app}\n")
        f.write(f"scenario={scenario}\n")
        f.write(f"duration_s={args.duration}\n")
        f.write(f"iface={args.iface}\n")
        f.write(f"netem={args.netem or 'none'}\n")
        f.write(f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    print(f"[capture_lab] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
