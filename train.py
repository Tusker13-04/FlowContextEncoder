"""
Clean modular training script for Phase 0.

Usage:
    python train.py

For Phase 1, replace `FlowDataset.from_synthetic()` with your real dataset loader.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import FlowContextEncoder, SupConLoss
from data.flow_dataset import FlowDataset
from configs.phase0_config import Phase0Config


def compute_similarity_stats(model, loader, device):
    """Compute mean intra-class and inter-class cosine similarities over a batch."""
    model.eval()
    with torch.no_grad():
        pkts, ctx, mask, lbl = next(iter(loader))
        pkts, ctx, mask = pkts.to(device), ctx.to(device), mask.to(device)
        z = model(pkts, ctx, mask)  # (B, d_emb)
        sim = z @ z.T               # (B, B)
        B = z.shape[0]
        intra, inter = [], []
        for i in range(B):
            for j in range(i + 1, B):
                if lbl[i] == lbl[j]:
                    intra.append(sim[i, j].item())
                else:
                    inter.append(sim[i, j].item())
    model.train()
    return (
        sum(intra) / len(intra) if intra else 0.0,
        sum(inter) / len(inter) if inter else 0.0,
    )


def main():
    cfg    = Phase0Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(cfg.train.seed)

    # Dataset
    dataset = FlowDataset.from_synthetic(
        n_classes=cfg.data.n_classes,
        n_per_class=cfg.data.n_per_class,
        max_len=cfg.data.max_len,
    )
    n_val  = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False, drop_last=False)

    # Model
    model = FlowContextEncoder(
        d_in=cfg.model.d_in,
        d_ctx=cfg.model.d_ctx,
        d_model=cfg.model.d_model,
        d_emb=cfg.model.d_emb,
        n_layers=cfg.model.n_layers,
        d_state=cfg.model.d_state,
        max_len=cfg.model.max_len,
        dropout=cfg.model.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = SupConLoss(temperature=cfg.train.temperature)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.train.epochs)

    # Training loop
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0
        for pkts, ctx, mask, lbl in train_loader:
            pkts, ctx, mask, lbl = pkts.to(device), ctx.to(device), mask.to(device), lbl.to(device)
            z    = model(pkts, ctx, mask)
            loss = criterion(z, lbl)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            intra, inter = compute_similarity_stats(model, val_loader, device)
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Intra sim: {intra:.4f} | Inter sim: {inter:.4f}")

    print("\nTraining complete.")
    torch.save(model.state_dict(), "flow_context_encoder_phase0.pt")
    print("Saved: flow_context_encoder_phase0.pt")


if __name__ == '__main__':
    main()
