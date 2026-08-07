"""
utils/train.py
───────────────
Training loop for MusicTransformer.

Trains from scratch using next-token prediction (cross-entropy).
Saves best checkpoint based on validation loss.
Logs training and validation loss to CSV.
"""

import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.losses import combined_loss, build_token_value_tensor


def get_lr_scheduler(optimizer, warmup_steps: int, d_model: int):
    """
    Learning rate schedule from "Attention is All You Need":
      lr = d_model^{-0.5} · min(step^{-0.5}, step · warmup^{-1.5})
    Increases linearly for warmup_steps, then decreases.
    """
    def lr_lambda(step):
        step = max(1, step)
        return (d_model ** -0.5) * min(
            step ** -0.5,
            step * (warmup_steps ** -1.5)
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, device,
                grad_clip: float, log_every: int, epoch: int,
                cfg: dict, token_values: torch.Tensor):
    """Run one epoch of training. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        logits = model(x)                          # (B, N, vocab_size)
        loss   = combined_loss(logits, y, cfg, token_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        if batch_idx % log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}"
            })

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch(model, loader, device, epoch: int,
               cfg: dict, token_values: torch.Tensor):
    """Run one epoch of evaluation. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = combined_loss(logits, y, cfg, token_values)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


def train(model, train_dataset, val_dataset, cfg: dict):
    """
    Full training loop.

    cfg keys used:
      batch_size, learning_rate, max_epochs, warmup_steps,
      grad_clip, save_dir, log_every, d_model
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    print(f"Attention:   {cfg.get('attention_type', 'gaussian')}")
    print(f"Loss type:   {cfg.get('loss_type', 'ce')}")
    print(f"Parameters:  {model.count_parameters():,}\n")

    token_values = build_token_value_tensor(device)

    model = model.to(device)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.get("batch_size", 16),
        shuffle     = True,
        num_workers = 0,          # set >0 for faster loading on Linux
        pin_memory  = device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.get("batch_size", 16),
        shuffle     = False,
        num_workers = 0,
        pin_memory  = device.type == "cuda"
    )

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.get("learning_rate", 1e-4),
        weight_decay = 0.01,
        betas        = (0.9, 0.98)
    )
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps = cfg.get("warmup_steps", 1000),
        d_model      = cfg.get("d_model", 128)
    )

    # Save directory + CSV log
    save_dir = cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "time_s"])

    best_val_loss = float("inf")
    max_epochs    = cfg.get("max_epochs", 100)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip    = cfg.get("grad_clip", 1.0),
            log_every    = cfg.get("log_every", 50),
            epoch        = epoch,
            cfg          = cfg,
            token_values = token_values,
        )
        val_loss = eval_epoch(
            model, val_loader, device, epoch,
            cfg          = cfg,
            token_values = token_values,
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{max_epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"({elapsed:.1f}s)")

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, f"{elapsed:.1f}"])

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "best.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "cfg":         cfg,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "cfg":         cfg,
            }, ckpt_path)

    print(f"\nTraining complete. Best val_loss = {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}/")
    print(f"Training log:         {log_path}")
