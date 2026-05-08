#!/usr/bin/env python3
"""
eval.py  — Evaluate a trained checkpoint.

Usage:
    python eval.py --checkpoint checkpoints/best.pt
    python eval.py --checkpoint checkpoints/best.pt --npz_val data/val.npz

Outputs:
    - Mean intra-class cosine similarity  (KPI: > 0.7)
    - Mean inter-class cosine similarity  (KPI: < 0.3)
    - k-NN accuracy on the val set
    - UMAP / t-SNE cluster plot saved to eval_output/embedding_plot.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import FlowContextEncoder
from data  import FlowDataset, collate_flows


APP_NAMES = ["Video", "Gaming", "VoIP", "Bulk", "XR"]


def parse_args():
    p = argparse.ArgumentParser(description="FlowContextEncoder evaluation")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--npz_val",      type=str, default=None)
    p.add_argument("--n_synth",      type=int, default=500)
    p.add_argument("--n_classes",    type=int, default=5)
    p.add_argument("--max_len",      type=int, default=128)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--knn_k",        type=int, default=5)
    p.add_argument("--plot",         action="store_true",
                   help="Save UMAP scatter plot (requires umap-learn + matplotlib)")
    p.add_argument("--out_dir",      type=str, default="eval_output")
    return p.parse_args()


@torch.no_grad()
def embed_dataset(model, loader, device):
    model.eval()
    zs, ls = [], []
    for packets, ctx, labels, mask in loader:
        packets, ctx, mask = packets.to(device), ctx.to(device), mask.to(device)
        zs.append(model(packets, ctx, mask).cpu())
        ls.append(labels)
    return torch.cat(zs).numpy(), torch.cat(ls).numpy()


def compute_kpi(Z: np.ndarray, L: np.ndarray):
    """Returns (intra_mean, inter_mean) cosine similarity."""
    Z_t = torch.from_numpy(Z)
    sim = (Z_t @ Z_t.T).numpy()
    intra, inter = [], []
    M = len(L)
    for i in range(M):
        for j in range(i + 1, M):
            (intra if L[i] == L[j] else inter).append(sim[i, j])
    return float(np.mean(intra)), float(np.mean(inter))


def knn_eval(Z_tr, L_tr, Z_va, L_va, k=5):
    Z_tr_t = torch.from_numpy(Z_tr)
    Z_va_t = torch.from_numpy(Z_va)
    sim = Z_va_t @ Z_tr_t.T
    topk = sim.topk(k, dim=1).indices
    preds = torch.from_numpy(L_tr)[topk].mode(dim=1).values.numpy()
    return float((preds == L_va).mean())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"]

    # Infer model config from state dict shapes
    d_model = state["pkt_embed.weight"].shape[0]
    d_in    = state["pkt_embed.weight"].shape[1]
    d_ctx   = state["film.mlp.0.weight"].shape[1]
    d_embed = state["proj_head.2.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("blocks.") and k.endswith(".norm.weight"))
    max_len  = state["pos_emb.weight"].shape[0]

    model = FlowContextEncoder(
        d_in=d_in, d_ctx=d_ctx, d_model=d_model,
        d_embed=d_embed, n_layers=n_layers, max_len=max_len,
    ).to(device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # Dataset
    ds = FlowDataset(
        npz_path=args.npz_val,
        max_len=args.max_len,
        n_synth=args.n_synth,
        n_classes=args.n_classes,
        seed=99,
    )
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=collate_flows, shuffle=False)

    Z, L = embed_dataset(model, loader, device)
    intra, inter = compute_kpi(Z, L)
    knn_acc = knn_eval(Z, L, Z, L, k=args.knn_k)

    print("\n── Embedding KPIs ────────────────────────")
    print(f"  Intra-class cosine sim : {intra:.4f}  (target > 0.70)  {'✓' if intra > 0.70 else '✗'}")
    print(f"  Inter-class cosine sim : {inter:.4f}  (target < 0.30)  {'✓' if inter < 0.30 else '✗'}")
    print(f"  {args.knn_k}-NN accuracy          : {knn_acc:.4f}")
    print("──────────────────────────────────────────")

    # Optional UMAP plot
    if args.plot:
        try:
            import umap
            import matplotlib.pyplot as plt

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            Z2 = reducer.fit_transform(Z)

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, args.n_classes))
            for cls in range(args.n_classes):
                mask = L == cls
                label = APP_NAMES[cls] if cls < len(APP_NAMES) else str(cls)
                ax.scatter(Z2[mask, 0], Z2[mask, 1], s=15, alpha=0.7,
                           color=colors[cls], label=label)
            ax.legend(fontsize=9)
            ax.set_title("FlowContextEncoder — UMAP of embeddings")
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            out_path = out_dir / "embedding_plot.png"
            fig.savefig(out_path, dpi=150)
            print(f"\nUMAP plot saved → {out_path}")
        except ImportError:
            print("[warn] umap-learn or matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
