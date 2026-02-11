"""
Training utilities for the StockFormer pipeline.

Functions:
    - get_optimizer: Returns optimizer based on name
    - run_one_epoch: Runs a single training or validation epoch
    - train_model: Full training loop with LR scheduler, gradient clipping,
                   early stopping, and best-model checkpointing
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
import pandas as pd


# =============================================================================
# Optimizer Selection
# =============================================================================

def get_optimizer(model, optimizer_name, lr, weight_decay=0.01):
    """
    Returns the optimizer for training.

    Args:
        model: PyTorch model
        optimizer_name: "adam", "adamw", or "sgd"
        lr: Learning rate
        weight_decay: Weight decay for AdamW (default: 0.01)

    Returns:
        PyTorch optimizer

    To add a new optimizer, add an elif block below.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
# Curriculum Learning Sampler
# =============================================================================

class CurriculumSampler:
    """
    Curriculum learning: start with easy samples, gradually add harder ones.

    Easy = large |return| (unambiguous direction).
    Hard = small |return| (noisy, near zero boundary).

    Args:
        dataset: The Subset (training split) to sample from
        full_dataset: The underlying StockSequenceDataset
        num_epochs: Total epochs (ramp reaches 100% by num_epochs // 2)
        start_fraction: Fraction of easiest samples to start with (default: 0.3)
    """

    def __init__(self, dataset, full_dataset, num_epochs, start_fraction=0.3):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.start_fraction = start_fraction

        # Get absolute target values for each sample in the subset
        abs_returns = []
        indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
        for i in indices:
            row_idx = full_dataset.indices[i]
            val = abs(float(full_dataset.df.loc[row_idx, full_dataset.target_col]))
            abs_returns.append(val)

        # Sort by |return| descending (easiest first)
        abs_returns = np.array(abs_returns)
        self.sorted_order = np.argsort(-abs_returns)
        self.total = len(self.sorted_order)

    def get_indices(self, epoch):
        """Get training indices for a given epoch (curriculum ramp)."""
        ramp_end = max(1, self.num_epochs // 2)
        fraction = min(1.0, self.start_fraction + (1.0 - self.start_fraction) * epoch / ramp_end)
        n = max(1, int(fraction * self.total))
        return self.sorted_order[:n].tolist()


# =============================================================================
# Single Epoch
# =============================================================================

def run_one_epoch(model, loader, optimizer, loss_fn, device, label_mode,
                  is_training=True, max_grad_norm=1.0, entropy_reg_weight=0.0):
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
        entropy_reg_weight: Weight for entropy regularization (default: 0.0 = disabled).
            Encourages diverse predictions by penalizing low-entropy output distributions.

    Returns:
        Dict with "loss", "accuracy", and "pred_entropy" keys
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    total_entropy = 0.0
    class_counts = None

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

            # Entropy regularization: penalize collapsed (low-entropy) predictions
            if entropy_reg_weight > 0 and label_mode != "regression":
                probs = F.softmax(outputs, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                loss = loss - entropy_reg_weight * entropy

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

            # Track prediction entropy and class distribution for collapse detection
            with torch.no_grad():
                probs = F.softmax(outputs, dim=-1)
                batch_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                total_entropy += batch_entropy.item() * batch_x.size(0)

                # Count predicted classes
                num_classes = outputs.size(-1)
                if class_counts is None:
                    class_counts = torch.zeros(num_classes, dtype=torch.long)
                for c in range(num_classes):
                    class_counts[c] += (preds == c).sum().item()

    # Compute averages
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else float("nan")
    avg_entropy = total_entropy / len(loader.dataset) if total > 0 else float("nan")

    result = {"loss": avg_loss, "accuracy": accuracy, "pred_entropy": avg_entropy}

    # Collapse detection: warn if model predicts only one class
    if class_counts is not None and total > 0:
        class_pcts = class_counts.float() / class_counts.sum()
        dominant_pct = class_pcts.max().item()
        result["dominant_class_pct"] = dominant_pct
        result["class_distribution"] = class_pcts.tolist()

    return result


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
    entropy_reg_weight=0.0,
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
        entropy_reg_weight: Entropy regularization weight (default: 0.0 = disabled)

    Returns:
        Dict with training history (epoch, train_loss, val_loss, train_acc, val_acc, lr)
    """
    base_lr = optimizer.param_groups[0]["lr"]

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
    consecutive_collapse = 0  # Track epochs where dominant class > 90%
    COLLAPSE_THRESHOLD = 0.90
    COLLAPSE_PATIENCE = 3

    if entropy_reg_weight > 0 and label_mode != "regression":
        print(f"Entropy regularization enabled: weight={entropy_reg_weight}")

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
            entropy_reg_weight=entropy_reg_weight,
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
        entropy_str = ""
        if label_mode != "regression" and "pred_entropy" in val_metrics:
            entropy_str = f"  Entropy={val_metrics['pred_entropy']:.3f}"
        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"TrainLoss={train_metrics['loss']:.4f}  "
            f"ValLoss={val_metrics['loss']:.4f}  "
            f"TrainAcc={train_metrics['accuracy']:.4f}  "
            f"ValAcc={val_metrics['accuracy']:.4f}  "
            f"LR={current_lr:.2e}{entropy_str}",
            flush=True,
        )

        # Collapse detection: warn and halt if sustained
        dominant_pct = val_metrics.get("dominant_class_pct", 0)
        if dominant_pct > 0.85 and label_mode != "regression":
            dist = val_metrics.get("class_distribution", [])
            print(f"  WARNING: Possible collapse — {dominant_pct:.1%} predictions "
                  f"in one class. Distribution: {[f'{p:.2f}' for p in dist]}")

        if dominant_pct > COLLAPSE_THRESHOLD and label_mode != "regression":
            consecutive_collapse += 1
            if consecutive_collapse >= COLLAPSE_PATIENCE:
                print(
                    f"  HALT: Model collapsed — dominant class > {COLLAPSE_THRESHOLD:.0%} "
                    f"for {COLLAPSE_PATIENCE} consecutive epochs. "
                    f"Stopping training to prevent wasted compute."
                )
                break
        else:
            consecutive_collapse = 0

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
        os.makedirs(os.path.dirname(model_out_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), model_out_path)
        print(f"Saved model to {model_out_path}")

    # Save training log
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_df = pd.DataFrame(history)
        log_df.to_csv(log_path, index=False)
        print(f"Saved training log to {log_path}")

    return history
