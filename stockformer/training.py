"""
Training utilities for the StockFormer pipeline.

Functions:
    - get_optimizer: Returns optimizer based on name
    - run_one_epoch: Runs a single training or validation epoch
    - train_model: Full training loop with LR scheduler, gradient clipping,
                   early stopping, and best-model checkpointing
"""

import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd


# =============================================================================
# Optimizer Selection
# =============================================================================

def get_optimizer(model, optimizer_name, lr):
    """
    Returns the optimizer for training.

    Args:
        model: PyTorch model
        optimizer_name: "adam", "adamw", or "sgd"
        lr: Learning rate

    Returns:
        PyTorch optimizer

    To add a new optimizer, add an elif block below.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)

    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)

    # -------------------------------------------------------------------------
    # Add new optimizers here. Example:
    # -------------------------------------------------------------------------
    # elif optimizer_name == "adagrad":
    #     return torch.optim.Adagrad(model.parameters(), lr=lr)

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available: adam, adamw, sgd, rmsprop"
        )


# =============================================================================
# Single Epoch
# =============================================================================

def run_one_epoch(model, loader, optimizer, loss_fn, device, label_mode,
                  is_training=True, max_grad_norm=1.0):
    """
    Run one epoch of training or validation.

    Args:
        model: PyTorch model
        loader: DataLoader for the data
        optimizer: Optimizer (can be None for validation)
        loss_fn: Loss function
        device: Device to run on ("cuda" or "cpu")
        label_mode: "regression", "binary", or "buckets"
        is_training: If True, run training; if False, run validation
        max_grad_norm: Max gradient norm for clipping (default: 1.0)

    Returns:
        Dict with "loss" and "accuracy" keys
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        # Move data to device
        batch_x = batch_x.to(device)

        if label_mode == "regression":
            batch_y = batch_y.float().to(device)
        else:
            batch_y = batch_y.long().to(device)

        # Forward pass
        if is_training:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)

        # Track metrics
        total_loss += loss.item() * batch_x.size(0)

        if label_mode != "regression":
            preds = outputs.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.numel()

    # Compute averages
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else float("nan")

    return {"loss": avg_loss, "accuracy": accuracy}


# =============================================================================
# Full Training Loop
# =============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    label_mode,
    num_epochs,
    model_out_path=None,
    log_path=None,
    patience=7,
    warmup_epochs=3,
    max_grad_norm=1.0,
    model_type="encoder",
):
    """
    Full training loop with cosine LR scheduler, warmup, gradient clipping,
    early stopping, and best-model checkpointing.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on ("cuda" or "cpu")
        label_mode: "regression", "binary", or "buckets"
        num_epochs: Number of epochs to train
        model_out_path: Path to save the best model checkpoint (optional)
        log_path: Path to save training log CSV (optional)
        patience: Early stopping patience (default: 7)
        warmup_epochs: Number of linear warmup epochs (default: 3)
        max_grad_norm: Max gradient norm for clipping (default: 1.0)

    Returns:
        Dict with training history (epoch, train_loss, val_loss, train_acc, val_acc, lr)
    """
    base_lr = optimizer.param_groups[0]["lr"]

    # Move loss function to device (needed for FocalLoss class weight buffers)
    if hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.to(device)

    # Cosine annealing scheduler (applied after warmup)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_state_dict = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Linear warmup
        if epoch <= warmup_epochs:
            warmup_lr = base_lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        current_lr = optimizer.param_groups[0]["lr"]

        # Training epoch
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            label_mode=label_mode,
            is_training=True,
            max_grad_norm=max_grad_norm,
        )

        # Validation epoch
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            loss_fn=loss_fn,
            device=device,
            label_mode=label_mode,
            is_training=False,
        )

        # Step scheduler after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        # Print progress
        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"TrainLoss={train_metrics['loss']:.4f}  "
            f"ValLoss={val_metrics['loss']:.4f}  "
            f"TrainAcc={train_metrics['accuracy']:.4f}  "
            f"ValAcc={val_metrics['accuracy']:.4f}  "
            f"LR={current_lr:.2e}",
            flush=True,
        )

        # Best model checkpointing + early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  -> New best val_loss: {best_val_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs). "
                    f"Best val_loss: {best_val_loss:.4f}"
                )
                break

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # Save model checkpoint
    if model_out_path is not None:
        torch.save(model.state_dict(), model_out_path)
        print(f"Saved model to {model_out_path}")

    # Save training log
    if log_path is not None:
        log_df = pd.DataFrame(history)
        log_df.to_csv(log_path, index=False)
        print(f"Saved training log to {log_path}")

    return history
