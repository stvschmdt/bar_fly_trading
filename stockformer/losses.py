"""
Loss functions for the StockFormer pipeline.

Anti-collapse design:
    - FocalLoss: down-weights easy majority-class examples
    - Class weighting: inverse-frequency balancing
    - Label smoothing: prevents overconfident single-class predictions
    - Entropy regularization: penalizes low-entropy output distributions
    - Ordinal penalty: for bucket models, penalizes predictions far from true bucket

Functions:
    - compute_class_weights: Compute inverse-frequency class weights from labels
    - get_loss_function: Returns the appropriate loss for a given label_mode
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Class Weight Computation
# =============================================================================

def compute_class_weights(labels, num_classes=None, smoothing=0.1):
    """
    Compute inverse-frequency class weights from training labels.

    Uses smoothed inverse frequency: w_i = N / (num_classes * (count_i + smooth * N))
    This prevents extreme weights for very rare classes while still
    upweighting minority classes significantly.

    Args:
        labels: Array-like of integer class labels
        num_classes: Number of classes (auto-detected if None)
        smoothing: Smoothing factor (default: 0.1)

    Returns:
        torch.FloatTensor of shape [num_classes] with normalized weights
    """
    labels = np.asarray(labels, dtype=int)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    N = len(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

    # Smoothed inverse frequency
    weights = N / (num_classes * (counts + smoothing * N))

    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()

    return torch.FloatTensor(weights)


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples.
    Combined with class weights, this strongly counteracts majority-class collapse.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0). Higher = more focus on hard examples.
        alpha: Class weighting tensor (default: None). Should be [num_classes].
        label_smoothing: Smooth targets to prevent overconfidence (default: 0.0).
        reduction: How to reduce the loss ("mean", "sum", or "none")

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Register alpha as buffer so it moves with .to(device)
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.FloatTensor(alpha)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model outputs (logits), shape [batch, num_classes]
            targets: Ground truth labels, shape [batch]

        Returns:
            Focal loss value
        """
        num_classes = inputs.size(-1)

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            # Create soft targets
            smooth = self.label_smoothing
            one_hot = F.one_hot(targets, num_classes).float()
            soft_targets = one_hot * (1 - smooth) + smooth / num_classes
            # Compute log-softmax and soft cross entropy
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(soft_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Focal term: (1 - p_t)^gamma
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction="none"))
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Ordinal-Aware Focal Loss
# =============================================================================

class OrdinalFocalLoss(nn.Module):
    """
    Ordinal-aware Focal Loss for bucket/ordinal classification.

    Combines Focal Loss with an Earth Mover's Distance (EMD) penalty.
    The EMD component compares predicted and true CDFs, naturally penalizing
    predictions that are farther from the true bucket more than adjacent misses.

    Args:
        gamma: Focal loss focusing parameter (default: 2.0)
        emd_weight: Weight for the EMD penalty term (default: 1.0)
        alpha: Per-class weights for focal component (default: None)
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: "mean", "sum", or "none"
    """

    def __init__(self, gamma=2.0, emd_weight=1.0, alpha=None, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.emd_weight = emd_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.FloatTensor(alpha)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [batch, num_classes]
            targets: Integer labels [batch]
        """
        num_classes = inputs.size(-1)

        # Focal loss component
        if self.label_smoothing > 0:
            one_hot = F.one_hot(targets, num_classes).float()
            smooth = self.label_smoothing
            soft_targets = one_hot * (1 - smooth) + smooth / num_classes
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(soft_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction="none"))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        # EMD component: compare predicted CDF vs true CDF
        probs = F.softmax(inputs, dim=-1)
        pred_cdf = torch.cumsum(probs, dim=-1)
        true_onehot = F.one_hot(targets, num_classes).float()
        true_cdf = torch.cumsum(true_onehot, dim=-1)
        emd = ((pred_cdf - true_cdf) ** 2).sum(dim=-1)

        # Combined loss
        combined = focal_loss + self.emd_weight * emd

        if self.reduction == "mean":
            return combined.mean()
        elif self.reduction == "sum":
            return combined.sum()
        else:
            return combined


# =============================================================================
# Entropy-Regularized Loss Wrapper
# =============================================================================

class EntropyRegularizedLoss(nn.Module):
    """
    Wraps any classification loss with an entropy regularization term.

    The entropy term penalizes the model when the BATCH-LEVEL class distribution
    has low entropy (i.e., the model predicts the same class for most samples).

    Total loss = base_loss - entropy_weight * H(mean_probs)

    where H(p) = -sum(p_i * log(p_i)) is the entropy of the average predicted
    probability across the batch. Maximizing this entropy encourages the model
    to spread predictions across classes rather than collapsing to one.

    Args:
        base_loss: The underlying loss function (e.g., FocalLoss)
        entropy_weight: Weight for the entropy regularization (default: 0.1)
    """

    def __init__(self, base_loss, entropy_weight=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.entropy_weight = entropy_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model outputs (logits), shape [batch, num_classes]
            targets: Ground truth labels, shape [batch]
        """
        # Base loss
        loss = self.base_loss(inputs, targets)

        # Entropy regularization on batch-level prediction distribution
        probs = F.softmax(inputs, dim=-1)
        mean_probs = probs.mean(dim=0)  # [num_classes]
        # Entropy of the average prediction (higher = more diverse)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
        # Subtract negative entropy (= add entropy) to encourage diversity
        loss = loss - self.entropy_weight * entropy

        return loss


# =============================================================================
# Loss Function Selection
# =============================================================================

def get_loss_function(
    label_mode,
    loss_name=None,
    class_weights=None,
    label_smoothing=0.0,
    focal_gamma=2.0,
    entropy_weight=0.0,
):
    """
    Returns the appropriate loss function for the given label mode.

    Anti-collapse strategy:
        - For "binary" and "buckets": uses FocalLoss with class weights,
          label smoothing, and optional entropy regularization
        - For "regression": uses HuberLoss (robust to outliers)

    Args:
        label_mode: "regression", "binary", or "buckets"
        loss_name: Optional override to use a specific loss function
        class_weights: Tensor of class weights (from compute_class_weights)
        label_smoothing: Label smoothing factor (default: 0.0)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        entropy_weight: Entropy regularization weight (default: 0.0)

    Returns:
        PyTorch loss function (nn.Module)
    """
    # If a specific loss name is provided, use that
    if loss_name is not None:
        key = loss_name
    else:
        key = label_mode

    # Select loss function based on key
    if key == "regression":
        return nn.HuberLoss(delta=1.0)

    elif key in ("binary", "buckets"):
        # Default: focal loss with class weights + label smoothing
        base_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )
        if entropy_weight > 0:
            return EntropyRegularizedLoss(base_loss, entropy_weight=entropy_weight)
        return base_loss

    elif key == "mse":
        return nn.MSELoss()

    elif key == "mae" or key == "l1":
        return nn.L1Loss()

    elif key == "huber":
        return nn.HuberLoss()

    elif key == "smooth_l1":
        return nn.SmoothL1Loss()

    elif key == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    elif key == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    elif key == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )

    elif key == "ordinal_focal":
        base_loss = OrdinalFocalLoss(
            gamma=focal_gamma,
            emd_weight=1.0,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )
        if entropy_weight > 0:
            return EntropyRegularizedLoss(base_loss, entropy_weight=entropy_weight)
        return base_loss

    else:
        raise ValueError(
            f"Unknown loss function: {key}. "
            f"Available: regression, binary, buckets, mse, mae, l1, huber, "
            f"smooth_l1, cross_entropy, label_smoothing, focal, ordinal_focal"
        )
