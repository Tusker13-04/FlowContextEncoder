# FlowContextEncoder

**Context-Aware Flow Embeddings for Adaptive AI-based Network Traffic Classification**

> Phase 0 — Modular PyTorch implementation: Mamba SSM (parallel scan) + FiLM context fusion + Masked GAP + Supervised Contrastive Loss

---

## Architecture

```
Packets (B, N, d_in)  +  Context (B, d_ctx)  +  Mask (B, N)
     │
     ├── pkt_embed   Linear(d_in → d_model)
     ├── pos_emb     Learnable positional embedding
     ├── MambaBlock  × n_layers    ← Parallel Associative Scan SSM
     │       └── FiLM  (after block 0)  ← network context fusion
     ├── ctx_proj    additive context injection
     ├── LayerNorm
     ├── Masked GAP  (variable-length safe)
     ├── MLP head
     └── L2-norm  →  Embedding (B, d_embed)  on unit hypersphere
```

### Key components

| File | What it does |
|---|---|
| `model/ssm.py` | Pure-PyTorch parallel associative scan (Blelloch tree, O(N log N)) |
| `model/mamba_block.py` | Full Mamba block: selective conv → ZOH discretisation → scan → gate |
| `model/film.py` | FiLM context modulation (γ, β MLP from context vector) |
| `model/encoder.py` | End-to-end `FlowContextEncoder` with masked GAP + projection head |
| `model/loss.py` | `SupConLoss` — supervised contrastive loss (Khosla et al., 2020) |
| `data/features.py` | Per-packet (d_in=6) and context (d_ctx=4) feature definitions + synthetic generator |
| `data/dataset.py` | `FlowDataset` (.npz or synthetic) + `collate_flows` with padding mask |
| `configs/default.yaml` | All hyper-parameters |
| `train.py` | Training loop with cosine LR schedule, checkpointing, k-NN eval |
| `eval.py` | KPI reporting (intra/inter cosine sim) + optional UMAP plot |

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train on synthetic data (smoke test)
python train.py

# 3. Evaluate checkpoint
python eval.py --checkpoint checkpoints/best.pt

# 4. With real .npz data
python train.py --npz_train data/train.npz --npz_val data/val.npz

# 5. UMAP visualisation (needs umap-learn + matplotlib)
python eval.py --checkpoint checkpoints/best.pt --plot
```

---

## KPIs (Phase 0 targets)

| Metric | Target | Measured |
|---|---|---|
| Intra-class cosine similarity | > 0.70 | — |
| Inter-class cosine similarity | < 0.30 | — |
| k-NN accuracy (val) | ≥ 90% | — |
| Inference latency (per flow) | < 100 ms | — |

---

## Feature Definitions

### Per-packet features (`d_in = 6`)
| Idx | Feature | Range |
|---|---|---|
| 0 | direction (client→server = 0) | {0, 1} |
| 1 | packet size / 1500 (MTU norm) | [0, 1] |
| 2 | log1p(inter-arrival time ms) | [0, ~6] |
| 3 | TCP flags / 64 | [0, 1] |
| 4 | is_quic | {0, 1} |
| 5 | position fraction t/(N-1) | [0, 1] |

### Context features (`d_ctx = 4`)
| Idx | Feature | Range |
|---|---|---|
| 0 | RTT ms / 500 | [0, 1] |
| 1 | jitter ms / 100 | [0, 1] |
| 2 | retransmit rate | [0, 1] |
| 3 | packet rate / 1000 | [0, 1] |

---

## Datasets

| Dataset | Use |
|---|---|
| Synthetic (built-in) | Smoke test, architecture validation |
| [5G Traffic Datasets](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets) | Supervised fine-tuning |
| [CESNET-QUIC22](https://zenodo.org/record/7962299) | Encrypted traffic pretraining |
| MAWI | Cross-network generalisation |
| Manual pcap capture | YouTube / Netflix / XR under real conditions |

---

## Roadmap

- [x] Phase 0 — Core architecture + synthetic smoke-test
- [ ] Phase 1 — CICFlowMeter / Zeek feature pipeline for real datasets
- [ ] Phase 2 — Pretrain on CESNET-QUIC22 (self-supervised SimCLR)
- [ ] Phase 3 — Supervised fine-tune on 5G + in-house labels
- [ ] Phase 4 — Few-shot evaluation on unseen apps (XR)
- [ ] Phase 5 — Distillation + ONNX export for <100 ms inference
