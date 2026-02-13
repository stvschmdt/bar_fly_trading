"""
Loss functions for the StockFormer pipeline.

To add a new loss function:
    1. Add an elif block in get_loss_function()
    2. If it's a custom loss, define the class in this file

Functions:
    - get_loss_function: Returns the appropriate loss for a given label_mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Loss Function Selection
# =============================================================================

def get_loss_function(label_mode, loss_name=None, class_weights=None, **kwargs):
    """
    Returns the appropriate loss function for the given label mode.

    Args:
        label_mode: "regression", "binary", or "buckets"
        loss_name: Optional override to use a specific loss function
        class_weights: Optional tensor of per-class weights (for CrossEntropyLoss/FocalLoss)

    Returns:
        PyTorch loss function (nn.Module)

    To add a new loss function, add an elif block below.
    """
    # If a specific loss name is provided, use that
    if loss_name is not None:
        key = loss_name
    else:
        key = label_mode

    # Select loss function based on key
    if key == "regression":
        return nn.MSELoss()

    elif key == "binary":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif key == "buckets":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif key == "mse":
        return nn.MSELoss()

    elif key == "mae" or key == "l1":
        return nn.L1Loss()

    elif key == "huber":
        return nn.HuberLoss()

    elif key == "smooth_l1":
        return nn.SmoothL1Loss()

    elif key == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif key == "label_smoothing":
        smoothing = kwargs.get("label_smoothing", 0.1)
        return nn.CrossEntropyLoss(label_smoothing=smoothing, weight=class_weights)

    elif key == "focal":
        gamma = kwargs.get("focal_gamma", 2.0)
        smoothing = kwargs.get("label_smoothing", 0.0)
        return FocalLoss(gamma=gamma, alpha=class_weights, label_smoothing=smoothing)

    elif key == "ordinal_focal":
        return OrdinalFocalLoss(gamma=2.0, emd_weight=1.0, alpha=class_weights)

    elif key == "logcosh":
        return LogCoshLoss()

    elif key == "directional_mse":
        return DirectionalMSE(direction_weight=kwargs.get("direction_weight", 3.0))

    elif key == "combined_regression":
        return CombinedRegressionLoss(direction_weight=kwargs.get("direction_weight", 3.0))

    elif key == "symmetric_ce":
        return SymmetricCrossEntropy(alpha=1.0, beta=0.5, class_weights=class_weights)

    elif key == "soft_ordinal":
        return SoftLabelOrdinalLoss(sigma=1.0)

    elif key == "coral":
        return CoralLoss()

    else:
        raise ValueError(
            f"Unknown loss function: {key}. "
            f"Available: regression, binary, buckets, mse, mae, l1, huber, "
            f"smooth_l1, cross_entropy, label_smoothing, focal, ordinal_focal, "
            f"logcosh, directional_mse, combined_regression, symmetric_ce, "
            f"soft_ordinal, coral"
        )


# =============================================================================
# Custom Loss Classes
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples.
    Useful when one class is much more common than others.

    Args:
        gamma: Focusing parameter (default: 2.0). Higher values focus more on hard examples.
        alpha: Class weighting (default: None). Can be a float or tensor of weights.
        reduction: How to reduce the loss ("mean", "sum", or "none")
        label_smoothing: Smoothing factor (default: 0.0). Prevents overconfident predictions.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model outputs (logits), shape [batch, num_classes]
            targets: Ground truth labels, shape [batch]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss for regression.

    Smooth approximation that behaves like MSE for small errors and MAE
    for large errors. Robust to outlier moves in financial returns.
    Twice differentiable everywhere (unlike Huber). No hyperparameter to tune.
    """

    def forward(self, inputs, targets):
        diff = inputs - targets
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


class DirectionalMSE(nn.Module):
    """
    Directional MSE Loss for regression.

    Penalizes predictions that get the SIGN wrong more heavily than
    magnitude errors. Getting direction right on big moves matters more
    than minimizing MSE on noise.

    Args:
        direction_weight: Extra penalty multiplier for sign errors (default: 3.0).
            Total weight on sign-wrong samples = 1 + direction_weight.
    """

    def __init__(self, direction_weight=3.0):
        super().__init__()
        self.direction_weight = direction_weight

    def forward(self, inputs, targets):
        mse = (inputs - targets) ** 2
        sign_wrong = (torch.sign(inputs) != torch.sign(targets)).float()
        weighted_mse = mse * (1.0 + self.direction_weight * sign_wrong)
        return weighted_mse.mean()


class CombinedRegressionLoss(nn.Module):
    """
    Combined regression loss: Log-Cosh + Directional MSE.

    Default regression loss. Combines outlier robustness
    (Log-Cosh) with directional accuracy incentive (DMSE).

    Args:
        logcosh_weight: Weight for Log-Cosh component (default: 0.5)
        dmse_weight: Weight for Directional MSE component (default: 0.5)
        direction_weight: DMSE direction penalty (default: 3.0)
    """

    def __init__(self, logcosh_weight=0.5, dmse_weight=0.5, direction_weight=3.0):
        super().__init__()
        self.logcosh = LogCoshLoss()
        self.dmse = DirectionalMSE(direction_weight=direction_weight)
        self.logcosh_weight = logcosh_weight
        self.dmse_weight = dmse_weight

    def forward(self, inputs, targets):
        return (self.logcosh_weight * self.logcosh(inputs, targets) +
                self.dmse_weight * self.dmse(inputs, targets))


class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross-Entropy (SCE) for noisy-label classification.

    Combines standard CE with Reverse CE. More robust to noisy labels,
    which is critical because financial direction labels near 0% return
    are essentially random.

    Args:
        alpha: Weight for standard CE (default: 1.0)
        beta: Weight for Reverse CE (default: 0.5)
        class_weights: Optional per-class weights
    """

    def __init__(self, alpha=1.0, beta=0.5, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction="none")
        probs = F.softmax(inputs, dim=-1)
        rce = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return (self.alpha * ce + self.beta * rce).mean()


class SoftLabelOrdinalLoss(nn.Module):
    """
    Soft Label loss for ordinal/bucket classification.

    Replaces hard one-hot targets with Gaussian kernel centered on the
    true class. Adjacent classes get non-zero probability mass, naturally
    encoding ordinal structure.

    Args:
        sigma: Std dev of the Gaussian kernel (default: 1.0).
            Smaller = sharper (more like hard labels), larger = softer.
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, inputs, targets):
        K = inputs.size(-1)
        indices = torch.arange(K, device=inputs.device).float()
        soft_targets = torch.exp(
            -0.5 * ((indices.unsqueeze(0) - targets.unsqueeze(1).float()) / self.sigma) ** 2
        )
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        log_probs = F.log_softmax(inputs, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


class CoralLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) loss for ordinal classification.

    Transforms K-class ordinal problem into K-1 binary subtasks:
    "is the return >= threshold_k?" for each threshold. Guarantees
    monotonic cumulative probabilities and prevents class collapse.

    Requires the model's bucket head to output K-1 logits (not K).
    See model.py coral_head.

    Reference: Cao, Mirjalili & Raschka (2020)
    """

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch, K-1] — one logit per ordinal threshold
            targets: [batch] — class indices 0..K-1
        """
        num_thresholds = logits.size(1)
        levels = torch.arange(num_thresholds, device=logits.device)
        binary_targets = (targets.unsqueeze(1) > levels.unsqueeze(0)).float()
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


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
        reduction: "mean", "sum", or "none"
    """

    def __init__(self, gamma=2.0, emd_weight=1.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.emd_weight = emd_weight
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [batch, num_classes]
            targets: Integer labels [batch]
        """
        num_classes = inputs.size(-1)

        # Focal loss component
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                focal_loss = self.alpha * focal_loss
            else:
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
