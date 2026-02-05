"""
Training utilities for the StockFormer pipeline.

Functions:
    - get_optimizer: Returns optimizer based on name
    - run_one_epoch: Runs a single training or validation epoch
    - train_model: Full training loop with history tracking
"""

import torch
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

def run_one_epoch(model, loader, optimizer, loss_fn, device, label_mode, is_training=True):
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
):
    """
    Full training loop with validation and history tracking.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on ("cuda" or "cpu")
        label_mode: "regression", "binary", or "buckets"
        num_epochs: Number of epochs to train
        model_out_path: Path to save the model checkpoint (optional)
        log_path: Path to save training log CSV (optional)

    Returns:
        Dict with training history (epoch, train_loss, val_loss, train_acc, val_acc)
    """
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        # Training epoch
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            label_mode=label_mode,
            is_training=True,
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

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        # Print progress
        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"TrainLoss={train_metrics['loss']:.4f}  "
            f"ValLoss={val_metrics['loss']:.4f}  "
            f"TrainAcc={train_metrics['accuracy']:.4f}  "
            f"ValAcc={val_metrics['accuracy']:.4f}",
            flush=True
        )

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
