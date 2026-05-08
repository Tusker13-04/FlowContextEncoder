# FlowContextEncoder — Project Roadmap

## Phase 0 · Repo Hygiene ✅
| Item | Status |
|---|---|
| `requirements.txt` full deps | ✅ |
| `configs/default.yaml` canonical hyperparams | ✅ |
| `configs/pretrain.yaml` self-supervised settings | ✅ |
| `configs/finetune.yaml` supervised contrastive settings | ✅ |
| `.github/workflows/ci.yml` pytest on push | ✅ |
| `ROADMAP.md` this file | ✅ |

## Phase 1 · Data Pipeline (Week 1) 🔄
**Goal:** raw pcap / CESNET parquet → clean `(packets, context, label)` tensors.

| Step | File | Status |
|---|---|---|
| 1.1 Flow builder | `scripts/build_flows.py` | 🔄 scaffold |
| 1.2 CESNET-QUIC22 loader | `data/cesnet_dataset.py` | 🔄 scaffold |
| 1.3 5G dataset loader | `data/fiveg_dataset.py` | 🔄 scaffold |
| 1.4 Manual capture pipeline | `scripts/capture_lab.py` | 🔄 scaffold |
| 1.5 Leakage-resistant splits | `scripts/make_splits.py` | 🔄 scaffold |

**Flow schema** (per-row in `data/flows/*.parquet`):
```
flow_id          str     unique identifier
app_label        str     coarse class: video_streaming | gaming | voip | web | xr
packets          array   shape (N=128, 5): [size, direction, iat, tcp_flags, quic_type]
rtt_ms           float   estimated RTT in milliseconds
jitter_ms        float   RTT jitter in milliseconds
pkt_loss_rate    float   packet loss fraction [0,1]
throughput_kbps  float   flow throughput in kbps
split            str     train | val | test | fewshot
```

## Phase 2 · Encoder Training (Week 2)
- [ ] Self-supervised contrastive pretraining on CESNET-TLS/QUIC + MAWI
- [ ] Supervised contrastive fine-tuning on 5G + in-house labeled flows
- [ ] ArcFace / SupCon loss comparison
- [ ] Embedding TSNE/UMAP visualisations

## Phase 3 · Classifier Integration (Week 3)
- [ ] k-NN classifier over embeddings (Faiss)
- [ ] Prototype network for few-shot new apps
- [ ] SVM baseline comparison
- [ ] Per-class cosine similarity report

## Phase 4 · Evaluation & KPIs
| KPI | Target | Status |
|---|---|---|
| Intra-class cosine similarity | > 0.70 | ⏳ |
| Inter-class cosine similarity | < 0.30 | ⏳ |
| Test accuracy (encrypted+unencrypted) | ≥ 90% | ⏳ |
| Few-shot generalization accuracy | ≥ 85% | ⏳ |
| Per-flow inference latency | < 100 ms | ⏳ |

## Phase 5 · Optimisation & Deployment
- [ ] ONNX export + quantisation
- [ ] TensorRT inference benchmark
- [ ] Optional: PacketCLIP-style text alignment for interpretability
