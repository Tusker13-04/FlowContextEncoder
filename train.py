#!/usr/bin/env python3
"""
train.py  — Phase-0 training entry point.

Usage:
    python train.py                         # synthetic data, default config
    python train.py --config configs/default.yaml
    python train.py --npz_train data/train.npz --npz_val data/val.npz
    python train.py --n_epochs 50 --lr 1e-3
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FlowContextEncoder, SupConLoss
from data  import FlowDataset, collate_flows


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FlowContextEncoder — Phase 0 training")

    # data
    p.add_argument("--npz_train",         type=str,   default=None)
    p.add_argument("--npz_val",           type=str,   default=None)
    p.add_argument("--n_synth_train",     type=int,   default=2000)
    p.add_argument("--n_synth_val",       type=int,   default=400)
    p.add_argument("--n_classes",         type=int,   default=5)
    p.add_argument("--max_len",           type=int,   default=128)
    p.add_argument("--batch_size",        type=int,   default=64)
    p.add_argument("--num_workers",       type=int,   default=0)

    # model
    p.add_argument("--d_in",              type=int,   default=6)
    p.add_argument("--d_ctx",             type=int,   default=4)
    p.add_argument("--d_model",           type=int,   default=64)
    p.add_argument("--d_embed",           type=int,   default=128)
    p.add_argument("--n_layers",          type=int,   default=3)
    p.add_argument("--d_state",           type=int,   default=16)
    p.add_argument("--d_conv",            type=int,   default=4)
    p.add_argument("--expand",            type=int,   default=2)
    p.add_argument("--proj_mult",         type=int,   default=2)

    # loss / optim / schedule
    p.add_argument("--temperature",       type=float, default=0.07)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--weight_decay",      type=float, default=1e-2)
    p.add_argument("--n_epochs",          type=int,   default=30)
    p.add_argument("--log_every",         type=int,   default=5)

    # misc
    p.add_argument("--save_dir",          type=str,   default="checkpoints")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--config",            type=str,   default=None,
                   help="Optional YAML config file (keys override defaults above)")
    return p.parse_args()


def load_yaml_config(path: str, args: argparse.Namespace) -> argparse.Namespace:
    """Merge a YAML config file into args (YAML values take precedence)."""
    try:
        import yaml
    except ImportError:
        print("[warn] PyYAML not installed — skipping config file.")
        return args
    with open(path) as f:
        cfg = yaml.safe_load(f)
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    for k, v in flat.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_similarity_stats(
    model: FlowContextEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute mean intra-class and inter-class cosine similarity on a dataset."""
    model.eval()
    all_z, all_labels = [], []
    for packets, ctx, labels, mask in loader:
        packets, ctx, mask = packets.to(device), ctx.to(device), mask.to(device)
        z = model(packets, ctx, mask)
        all_z.append(z.cpu())
        all_labels.append(labels)

    Z = torch.cat(all_z)           # (M, d_embed)
    L = torch.cat(all_labels)      # (M,)
    sim_matrix = Z @ Z.T           # cosine sim (already L2-normed)

    intra, inter = [], []
    M = Z.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            s = sim_matrix[i, j].item()
            if L[i] == L[j]:
                intra.append(s)
            else:
                inter.append(s)

    return {
        "intra_mean": float(np.mean(intra)) if intra else 0.0,
        "inter_mean": float(np.mean(inter)) if inter else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# k-NN accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def knn_accuracy(
    model: FlowContextEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    k: int = 5,
) -> float:
    model.eval()

    def embed(loader):
        zs, ls = [], []
        for packets, ctx, labels, mask in loader:
            packets, ctx, mask = packets.to(device), ctx.to(device), mask.to(device)
            zs.append(model(packets, ctx, mask).cpu())
            ls.append(labels)
        return torch.cat(zs), torch.cat(ls)

    Z_tr, L_tr = embed(train_loader)
    Z_va, L_va = embed(val_loader)

    sim = Z_va @ Z_tr.T          # (n_val, n_train)
    topk = sim.topk(k, dim=1).indices  # (n_val, k)
    preds = L_tr[topk].mode(dim=1).values
    acc = (preds == L_va).float().mean().item()
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = FlowDataset(
        npz_path=args.npz_train,
        max_len=args.max_len,
        n_synth=args.n_synth_train,
        n_classes=args.n_classes,
        seed=args.seed,
    )
    val_ds = FlowDataset(
        npz_path=args.npz_val,
        max_len=args.max_len,
        n_synth=args.n_synth_val,
        n_classes=args.n_classes,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_flows, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_flows, num_workers=args.num_workers,
    )
    print(f"Train : {len(train_ds):,} flows  |  Val : {len(val_ds):,} flows")

    # ── Model ────────────────────────────────────────────────────────────────
    model = FlowContextEncoder(
        d_in=args.d_in,
        d_ctx=args.d_ctx,
        d_model=args.d_model,
        d_embed=args.d_embed,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        max_len=args.max_len,
        proj_mult=args.proj_mult,
    ).to(device)
    print(f"Parameters : {model.num_parameters:,}")

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────────────
    criterion = SupConLoss(temperature=args.temperature)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.n_epochs, eta_min=1e-6,
    )

    # ── Checkpointing ────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = math.inf

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for packets, ctx, labels, mask in train_loader:
            packets = packets.to(device)
            ctx     = ctx.to(device)
            labels  = labels.to(device)
            mask    = mask.to(device)

            optimiser.zero_grad()
            z    = model(packets, ctx, mask)
            loss = criterion(z, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Checkpoint best model
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "loss": avg_loss},
                save_dir / "best.pt",
            )

        if epoch % args.log_every == 0 or epoch == 1:
            stats = compute_similarity_stats(model, val_loader, device)
            acc   = knn_accuracy(model, train_loader, val_loader, device, k=5)
            print(
                f"Epoch {epoch:03d}/{args.n_epochs} "
                f"| loss {avg_loss:.4f} "
                f"| intra {stats['intra_mean']:.4f} "
                f"| inter {stats['inter_mean']:.4f} "
                f"| 5-NN acc {acc:.3f}"
            )

    print(f"\nTraining complete. Best checkpoint → {save_dir / 'best.pt'}")
    torch.save(
        {"epoch": args.n_epochs, "model_state": model.state_dict()},
        save_dir / "last.pt",
    )


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        args = load_yaml_config(args.config, args)
    train(args)
