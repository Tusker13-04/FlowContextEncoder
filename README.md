# FlowContextEncoder

> **Context-Aware Flow Embeddings for Adaptive AI-based Network Traffic Classification**  
> Phase 0 — Architecture & Prototype (AX Hackathon 2026)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem

Traditional network traffic classifiers (DPI, port-based, static rules) fail on:
- **Encrypted flows** (TLS 1.3 / QUIC / HTTP3) — no payload to inspect.
- **Dynamic / new app types** (XR, cloud gaming, serverless APIs) — rules go stale immediately.
- **Changing network conditions** — the same app behaves differently at high RTT / jitter.

The goal: learn a **flow embedding function** `f(packets, network_context) → z ∈ ℝ¹²⁸` such that:
- `cosine_sim(z_YouTube, z_Netflix) > 0.7` (same traffic class → close)
- `cosine_sim(z_YouTube, z_Gaming) < 0.3` (different class → far)
- Classification accuracy ≥ 90% on encrypted + unencrypted flows.
- Inference latency < 100 ms per flow.

---

## Phase 0 Architecture

### Overview

```
Input
  packets: (B, N, d_in)        # B flows, N≤128 packets each, d_in=8 features per packet
  context: (B, d_ctx)          # RTT, jitter, loss-rate, throughput  (d_ctx=4)
  mask:    (B, N)              # True = real packet, False = padding

Pipeline
  1. Packet Embedding         Linear(d_in → d_model)  +  positional encoding
  2. Mamba Block 0            Selective SSM (parallel scan) + causal Conv1d gate
  3. FiLM Context Fusion      γ(context) ⊙ x + β(context)   — injects RTT/jitter
  4. Mamba Blocks 1…n-1       Stack of selective SSM layers
  5. Context Residual         x = x + proj(context).unsqueeze(1)  — late reinforcement
  6. Masked GAP               Mean-pool over non-padded positions only
  7. MLP Projection Head      (d_model → d_model) → ReLU → (d_model → d_emb)
  8. L2 Normalisation         ‖z‖₂ = 1  — cosine similarity = dot product

Output
  z: (B, 128)   — unit-norm flow embedding
```

---

### 1. Input Feature Set

| Feature | Dim | Notes |
|---|---|---|
| Packet direction | 1 | 1 = outbound, -1 = inbound |
| Packet size | 1 | bytes, normalised by 1500 |
| Inter-arrival time | 1 | seconds, log-scaled |
| TCP flag bits | 1 | encoded as float (SYN, ACK, FIN, RST, PSH) |
| Cumulative bytes | 1 | running total, normalised |
| Burst marker | 1 | 1 if IAT < 1ms burst threshold |
| Size normalised | 1 | z-score within flow |
| Position | 1 | index / N |

Network context vector `(d_ctx=4)`: RTT estimate (ms), jitter (ms), packet-loss rate (%), normalised throughput (Mbps).

---

### 2. Packet Embedding + Positional Encoding

A linear projection `d_in → d_model` followed by **sinusoidal positional encoding** (no learned PE — avoids overfitting to fixed flow lengths):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Dropout (p=0.1) after adding PE — standard regularisation before entering deep layers.

---

### 3. MambaBlock — Selective State Space Model

Each MambaBlock processes a sequence `(B, N, d_model)` and returns the same shape.

#### 3a. Selective Projections

Unlike a fixed SSM (e.g., S4), **Mamba makes A, B, C input-dependent per timestep**:

```
x, z  = split( in_proj(u), dim=-1 )           # d_model → 2×d_inner
x     = causal_conv1d(x)                      # local context (k=4)
B_t   = linear_B(x_proj(x))                  # (B, N, d_state) — input-dependent
C_t   = linear_C(x_proj(x))                  # (B, N, d_state) — input-dependent
Δ_t   = softplus( dt_proj( linear_dt(x) ) )  # (B, N, d_inner) — time step, always > 0
```

This **selectivity** is why Mamba outperforms fixed-parameter RNNs on irregular sequences like packet flows — it can focus on or ignore packets based on their content.

#### 3b. Discretisation (Zero-Order Hold)

```
A_log  ∈ ℝ^(d_inner × d_state)   — learnable log-decay, fixed across time
Ã_t   = exp( Δ_t ⊗ A_log )       # (B, N, d_inner, d_state)  — continuous → discrete
B̃_t   = Δ_t ⊗ B_t               # (B, N, d_inner, d_state)
```

The ZOH discretisation ensures the state equation `h_t = Ã_t h_{t-1} + B̃_t x_t` is stable and matches continuous-time SSM theory.

#### 3c. Parallel Associative Scan

The recurrence `h_t = A_t h_{t-1} + B_t x_t` is a **linear prefix sum** over the sequence dimension. The parallel scan computes all N states simultaneously:

1. **Pad** sequence to next power of 2.
2. **Up-sweep:** compose adjacent (A, B) pairs up a binary tree — `O(N)` multiplications at each of `O(log N)` levels.
3. **Down-sweep:** distribute prefix result back down — all positions receive their inclusive prefix.
4. **Trim** back to original length N.

Complexity: `O(N log N)` vs `O(N)` serial — roughly equal for N=128, but parallelises fully on GPU.

Composition rule for two consecutive linear maps:
```
(A₂, B₂) ∘ (A₁, B₁)  =  (A₂ · A₁,   A₂ · B₁ + B₂)
```

#### 3d. Output and Gating

```
y_t  = (h_t * C_t).sum(-1)     # read out from state via input-dependent C
y    = out_proj( y * silu(z) ) # SiLU gate suppresses irrelevant channels
output = u + y                  # residual connection preserves input information
```

---

### 4. FiLM Context Fusion (after Block 0)

**Feature-wise Linear Modulation** conditions every position in the sequence on the network context vector without modifying the SSM structure:

```
[γ, β] = MLP(context)     # shape: (B, 2×d_model)
x      = (1 + γ) ⊙ x + β  # broadcast over N
```

Why FiLM after Block 0 (not before)?
- Block 0 builds a rich local representation from raw packets.
- FiLM then modulates *that representation* with network context — high-RTT flows will activate different channels than low-RTT flows of the same app.
- Placing it mid-stack (not at the input) avoids the context overwhelming the early packet features.

---

### 5. Masked Global Average Pooling

Flows have variable lengths (16–256 real packets, padded to N=128).

```python
pooled = (x * mask.float().unsqueeze(-1)).sum(1) / mask.float().sum(1).clamp(min=1)
```

- Zero-pads do **not** contribute to the mean.
- Fully differentiable — no special casing in backward pass.
- More robust than CLS-token pooling for variable-length flows.

---

### 6. MLP Projection Head

```
d_model → d_model → ReLU → Dropout(0.1) → d_emb(128)
```

Follows SimCLR / supervised contrastive learning practice: a 2-layer MLP projection head produces the embedding that the loss function operates on. The head is **kept attached at test time** (unlike SimCLR where it is discarded) because the loss is supervised — the projections are already semantically aligned.

---

### 7. Supervised Contrastive Loss

All flows with the same application label are **multiple positives** for each anchor:

```
L = -1/|P(i)| · Σ_{p∈P(i)} log [ exp(z_i · z_p / τ) / Σ_{k≠i} exp(z_i · z_k / τ) ]
```

- Temperature `τ = 0.07` — sharp similarity discrimination.
- Anchors with no positive in the batch are skipped (`has_pos` guard).
- Batch size ≥ 64 with balanced class sampling recommended for stable gradients.

---

### 8. Downstream Classifier (k-NN)

After training the encoder:
1. **Encode** all labeled flows → embedding matrix.
2. **Index** with FAISS (or `sklearn.NearestNeighbors`) for fast ANN search.
3. **Classify** new flow by majority vote of `k=5` nearest neighbours.
4. **Add new class** by embedding a handful of labeled flows and inserting into the index — no retraining needed.

---

## PPT-Ready Architecture Bullet Points

### Encoder Model
- **Selective SSM (Mamba-style)** processes each packet flow as a variable-length sequence of up to 128 packets.
- Input features per packet: direction, size, inter-arrival time, TCP flags, cumulative bytes, burst marker, position (8 features total).
- Network context (RTT, jitter, loss rate, throughput) is fused mid-stack via **FiLM conditioning** — γ and β shift the entire sequence based on current path state.
- **Parallel Associative Scan** computes all N hidden states simultaneously using a Blelloch tree reduction — no Python loops over sequence length, fully GPU-parallelisable.
- **Masked Global Average Pooling** aggregates variable-length flows cleanly, ignoring zero-padded positions.
- **L2-normalised 128-dim embedding** output — cosine similarity equals dot product.

### Contrastive Learning
- **Supervised Contrastive Loss** with temperature τ = 0.07 — YouTube and Netflix flows are both positives for each other; gaming flows are negatives.
- Self-supervised augmentations for pretraining: time-shift packet IATs, scale RTT, drop random packets, resample flow segments — each augmentation simulates a realistic network impairment.
- FiLM conditioning means the same app under good and bad network conditions maps to nearby embeddings (context normalises the representation).

### Classifier Integration
- **k-NN classifier (k=5)** over L2-normalised embeddings — no retraining to add new classes.
- FAISS index for <1 ms ANN lookup across millions of flow embeddings.
- Optional: small MLP fine-tuned on top of frozen encoder for highest accuracy.
- **Few-shot generalisation**: XR or new app type added with as few as 10–20 labeled flows by computing their embeddings and updating the index.

### KPIs Addressed
- **Intra-class cosine similarity > 0.7**: enforced by SupConLoss pulling same-type embeddings together.
- **Inter-class cosine similarity < 0.3**: enforced by SupConLoss pushing different-type embeddings apart.
- **≥ 90% classification accuracy**: targeted via CESNET-QUIC22 + 5G traffic dataset fine-tuning.
- **< 100 ms per-flow inference**: ~480 K parameter encoder; ONNX export + INT8 quantization planned for Phase 1.

---

## Repository Structure

```
FlowContextEncoder/
├── flow_context_encoder.py   # Full self-contained prototype (encoder + loss + training loop)
├── model/
│   ├── __init__.py
│   ├── mamba_block.py        # MambaBlock with parallel scan
│   ├── encoder.py            # FlowContextEncoder (full model)
│   └── losses.py             # SupConLoss
├── configs/
│   └── phase0_config.py      # All hyperparameters in one place
├── data/
│   └── flow_dataset.py       # FlowDataset + synthetic data generator
├── train.py                  # Clean training script
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/Tusker13-04/FlowContextEncoder
cd FlowContextEncoder
pip install -r requirements.txt

# Run self-contained prototype (synthetic data, 15 epochs)
python flow_context_encoder.py

# Run modular training script
python train.py
```

---

## Suggested Datasets (Phase 1)

| Dataset | Protocol | Use |
|---|---|---|
| [CESNET-QUIC22](https://zenodo.org/record/7189254) | QUIC | Pretraining (153M flows, 102 classes) |
| [CESNET-TLS22](https://zenodo.org/record/7965515) | TLS | Pretraining + fine-tune (141M flows) |
| [5G Traffic](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets) | Mixed | Fine-tuning (gaming / streaming) |
| MAWI Archive | Mixed | Cross-network generalisation |
| Manual Captures | TLS/QUIC | Lab evaluation (YouTube/Netflix/XR) |

---

## License

MIT
