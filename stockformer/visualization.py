"""
Visualization utilities for the StockFormer pipeline.

Functions:
    - plot_training_curves: Plot loss and accuracy over epochs
    - plot_training_curves_from_log: Plot from a log CSV file
"""

import os

import pandas as pd


def plot_training_curves(history, output_path="training_curves.png", show=False):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        output_path: Path to save the plot (PNG)
        show: If True, display the plot interactively

    Returns:
        Path to saved plot, or None if plotting failed
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')  # Non-interactive backend for saving
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot generation")
        return None

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    # Check if we have accuracy data (not NaN)
    has_accuracy = not all(pd.isna(train_acc)) and not all(pd.isna(val_acc))

    if has_accuracy:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        axes = [axes]

    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else None

    # Colors
    train_color = '#2ecc71'  # Green
    val_color = '#e74c3c'    # Red

    # -------------------------------------------------------------------------
    # Loss Plot
    # -------------------------------------------------------------------------
    ax_loss = axes[0]
    ax_loss.plot(epochs, train_loss, color=train_color, linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax_loss.plot(epochs, val_loss, color=val_color, linewidth=2, label='Val Loss', marker='s', markersize=4)

    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax_loss.legend(loc='upper right', fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    # Add min val loss annotation
    min_val_idx = val_loss.index(min(val_loss))
    min_val = min(val_loss)
    ax_loss.annotate(
        f'Best: {min_val:.4f}',
        xy=(epochs[min_val_idx], min_val),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=9,
        color=val_color,
        arrowprops=dict(arrowstyle='->', color=val_color, lw=0.5)
    )

    # -------------------------------------------------------------------------
    # Accuracy Plot (if available)
    # -------------------------------------------------------------------------
    if has_accuracy:
        ax_acc = axes[1]

        # Filter out NaN values for plotting
        valid_train_acc = [a for a in train_acc if not pd.isna(a)]
        valid_val_acc = [a for a in val_acc if not pd.isna(a)]
        valid_epochs = epochs[:len(valid_train_acc)]

        ax_acc.plot(valid_epochs, valid_train_acc, color=train_color, linewidth=2, label='Train Acc', marker='o', markersize=4)
        ax_acc.plot(valid_epochs, valid_val_acc, color=val_color, linewidth=2, label='Val Acc', marker='s', markersize=4)

        ax_acc.set_xlabel('Epoch', fontsize=12)
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        ax_acc.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax_acc.legend(loc='lower right', fontsize=10)
        ax_acc.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Add max val accuracy annotation
        if valid_val_acc:
            max_val_idx = valid_val_acc.index(max(valid_val_acc))
            max_val = max(valid_val_acc)
            ax_acc.annotate(
                f'Best: {max_val:.1%}',
                xy=(valid_epochs[max_val_idx], max_val),
                xytext=(10, -10),
                textcoords='offset points',
                fontsize=9,
                color=val_color,
                arrowprops=dict(arrowstyle='->', color=val_color, lw=0.5)
            )

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved training curves to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return output_path


def plot_training_curves_from_log(log_path, output_path=None, show=False):
    """
    Plot training curves from a log CSV file.

    Args:
        log_path: Path to training log CSV
        output_path: Path to save plot (default: same dir as log, .png extension)
        show: If True, display the plot interactively

    Returns:
        Path to saved plot, or None if plotting failed
    """
    if not os.path.exists(log_path):
        print(f"[WARN] Log file not found: {log_path}")
        return None

    df = pd.read_csv(log_path)

    # Convert to history dict format
    history = {
        "epoch": df["epoch"].tolist(),
        "train_loss": df["train_loss"].tolist(),
        "val_loss": df["val_loss"].tolist(),
        "train_acc": df["train_acc"].tolist() if "train_acc" in df.columns else [float('nan')] * len(df),
        "val_acc": df["val_acc"].tolist() if "val_acc" in df.columns else [float('nan')] * len(df),
    }

    # Default output path
    if output_path is None:
        base = os.path.splitext(log_path)[0]
        output_path = f"{base}_curves.png"

    return plot_training_curves(history, output_path, show)


def plot_loss_comparison(log_paths, labels=None, output_path="loss_comparison.png"):
    """
    Compare loss curves from multiple training runs.

    Args:
        log_paths: List of paths to training log CSVs
        labels: List of labels for each run (default: filenames)
        output_path: Path to save the plot

    Returns:
        Path to saved plot, or None if plotting failed
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot generation")
        return None

    if labels is None:
        labels = [os.path.basename(p).replace('.csv', '') for p in log_paths]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        if not os.path.exists(log_path):
            print(f"[WARN] Skipping missing log: {log_path}")
            continue

        df = pd.read_csv(log_path)
        color = colors[i % len(colors)]

        ax.plot(df["epoch"], df["val_loss"], color=color, linewidth=2, label=f'{label}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved loss comparison to {output_path}")
    return output_path
