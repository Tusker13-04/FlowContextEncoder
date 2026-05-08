# Data — Download & Preparation Guide

This directory holds **processed flow data** (parquet files) and **loader scripts**.
Raw pcaps and CSV files should go in `data/raw/` which is gitignored.

---

## Flow Schema

Every parquet file in `data/flows/` follows this schema:

| Column | Type | Description |
|---|---|---|
| `flow_id` | str | MD5-based unique flow identifier |
| `app_label` | str | Coarse class: `video_streaming \| gaming \| voip \| web \| xr` |
| `packets` | ndarray (128, 5) | Per-packet features: `[size, direction, iat, tcp_flags, quic_type]` |
| `rtt_ms` | float32 | Estimated RTT in milliseconds |
| `jitter_ms` | float32 | RTT/IAT jitter in milliseconds |
| `pkt_loss_rate` | float32 | Packet loss fraction `[0, 1]` |
| `throughput_kbps` | float32 | Flow throughput in kbps |
| `split` | str | `train \| val \| test \| fewshot` |

---

## Dataset 1 — CESNET-QUIC22

**Description:** 4 weeks of QUIC traffic from a Czech ISP backbone (~153M flows, 102 app classes).  
**Best for:** Encrypted QUIC traffic pretraining and domain recognition.

### Download

```bash
# Install DataZoo
pip install cesnet-datazoo

# The library handles download automatically on first use:
python - <<'EOF'
from cesnet_datazoo.datasets import CESNET_QUIC22
from cesnet_datazoo.config import DatasetConfig
cfg = DatasetConfig(data_root="data/cesnet")
dataset = CESNET_QUIC22(cfg)  # triggers download
EOF
```

### Convert to project parquet

```bash
python data/cesnet_dataset.py  # streams → data/flows/cesnet_*.parquet
```

Or stream directly from Python:
```python
from data.cesnet_dataset import stream_to_parquet
stream_to_parquet("CESNET-QUIC22", output_dir="data/flows", max_flows=500_000)
```

---

## Dataset 2 — 5G Traffic Datasets

**Description:** Labelled 5G traffic traces (gaming, streaming, etc.).  
**Source:** https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets

### Download

```bash
# Requires Kaggle API key in ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d kimdaegyeom/5g-traffic-datasets -p data/raw/5g/ --unzip
```

### Convert to project parquet

```bash
python data/fiveg_dataset.py --csv-dir data/raw/5g/ --output data/flows/ --split train
```

---

## Dataset 3 — Manual Lab Captures

**Description:** Your own pcaps of YouTube, Netflix, gaming, XR apps  
under good/bad network conditions (tc-netem impairment).  
**Best for:** Controlled ground-truth with known app + network scenario labels.

### Capture

```bash
# Good network — 120s YouTube capture
sudo python scripts/capture_lab.py \
    --label video_streaming --split train --iface eth0 --duration 120

# Bad network — 200ms RTT, 5% loss, 120s gaming capture
sudo python scripts/capture_lab.py \
    --label gaming --split test --iface eth0 --duration 120 \
    --netem "delay 200ms 20ms loss 5%"
```

---

## Dataset 4 — CESNET-TLS22 (optional, pretraining)

```bash
python data/cesnet_dataset.py  # change DatasetClass to CESNET_TLS22 in the script
```

---

## Build Leakage-Resistant Splits

After populating `data/flows/`, run:

```bash
python scripts/make_splits.py \
    --flow-dir data/flows/ \
    --out-dir  splits/ \
    --strategy time \
    --fewshot-apps xr
```

This writes:
```
splits/train.txt
splits/val.txt
splits/test.txt
splits/fewshot.txt
splits/fewshot_apps.txt
```

---

## Directory Layout (after setup)

```
data/
├── README.md              ← this file
├── cesnet_dataset.py      ← CESNET DataZoo loader
├── fiveg_dataset.py       ← 5G Kaggle loader
├── flows/                 ← processed parquet files (gitignored)
│   ├── cesnet_quic22_train_chunk0000.parquet
│   ├── 5g_traffic_train.parquet
│   └── video_streaming_train_youtube.parquet
├── raw/                   ← raw pcaps and CSVs (gitignored)
│   ├── 5g/
│   └── cesnet/
└── __init__.py
```

> **Note:** `data/flows/` and `data/raw/` are in `.gitignore`. Never commit raw traffic data.
