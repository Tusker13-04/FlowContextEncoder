# FlowContextEncoder

**Context-Aware Flow Embeddings for Adaptive AI-based Network Traffic Classification**

A PyTorch implementation of a packet-flow encoder that generates context-aware embeddings using flow-level features and network characteristics (RTT, jitter, path state) to enable accurate traffic classification in real-time and under changing conditions.

---

## Architecture

```
Packets (B, N, d_in)
  → pkt_embed  (Linear)
  → pos_enc    (sinusoidal)
  → MambaBlock[0]  ──── FiLM context fusion  ←── Context (B, d_ctx)
  → MambaBlock[1..n]
  → + ctx_proj  (additive global context residual)
  → Masked GAP  (handles variable-length flows)
  → MLP Head
  → L2-norm
  → Embedding (B, 128)
```

### Key Components

| Component | Description |
|---|---|
| **Parallel Associative Scan** | Blelloch tree scan — computes all N SSM hidden states in O(N log N); pure PyTorch, Windows-compatible, no custom CUDA |
| **MambaBlock** | Selective SSM with input-dependent B/C/Δ projections + causal conv1d + SiLU gating |
| **FiLM Fusion** | Feature-wise Linear Modulation: injects network context (RTT, jitter) after the first Mamba block via γ/β MLPs |
| **Masked GAP** | Handles variable-length flows; ignores zero-padded positions during pooling |
| **SupConLoss** | Supervised Contrastive Loss (τ=0.07): similar app-type flows cluster together, dissimilar types separate |

---

## Target KPIs

| Metric | Target |
|---|---|
| Intra-class cosine similarity | > 0.7 (e.g. YouTube ↔ Netflix) |
| Inter-class cosine similarity | < 0.3 (e.g. video streaming ↔ gaming) |
| Classification accuracy | ≥ 90% on encrypted + unencrypted flows |
| Few-shot generalization | ≥ 85% on unseen app types |
| Inference latency | < 100 ms per flow |

---

## Quickstart

```bash
pip install torch
python flow_context_encoder.py
```

Expected output (15 epochs, synthetic data, CPU):
```
Device: cpu
Parameters: ~480,000
Epoch 01 | Loss: 4.17 | Intra sim: 0.12 | Inter sim: 0.08
...
Epoch 15 | Loss: 2.40 | Intra sim: 0.48 | Inter sim: 0.05
Output shape : torch.Size([4, 128])
L2 norms     : [1.0, 1.0, 1.0, 1.0]
```

---

## Input Features

### Per-packet (d_in = 9 default)
- Direction (0/1)
- Packet size
- Inter-arrival time
- TCP flags / QUIC packet type (one-hot or scalar)
- Relative position in flow

### Network context (d_ctx = 8 default)
- RTT (SYN–ACK, QUIC handshake)
- Jitter (RTT variation)
- Retransmission rate
- Path congestion signal
- Throughput estimates (uplink/downlink)
- Flow duration

---

## Datasets

| Dataset | Use |
|---|---|
| [CESNET-QUIC22](https://zenodo.org/record/7118539) | Pretraining on encrypted QUIC flows |
| [CESNET-TLS22](https://zenodo.org/record/6412094) | Encrypted TLS classification |
| [5G Traffic Datasets](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets) | Labeled gaming/streaming/5G flows |
| MAWI Archive | Cross-network generalization |
| Manual captures | YouTube/Netflix/gaming under variable RTT/jitter |

---

## Hackathon Context

This project was built for **AX Hackathon Phase 1** — problem statement: *"Context-Aware Flow Embeddings for Adaptive AI-based Network Traffic Classification"*.

---

## Citation / References

- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
- Pereira et al., "Self-Supervised Flow Embeddings", 2023
- CESNET-QUIC22 dataset: Luxemburk & Cejka, 2022
- PacketCLIP: multi-modal network traffic embedding, 2025
