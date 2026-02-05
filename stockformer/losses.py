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

def get_loss_function(label_mode, loss_name=None):
    """
    Returns the appropriate loss function for the given label mode.

    Args:
        label_mode: "regression", "binary", or "buckets"
        loss_name: Optional override to use a specific loss function

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
        return nn.CrossEntropyLoss()

    elif key == "buckets":
        return nn.CrossEntropyLoss()

    elif key == "mse":
        return nn.MSELoss()

    elif key == "mae" or key == "l1":
        return nn.L1Loss()

    elif key == "huber":
        return nn.HuberLoss()

    elif key == "smooth_l1":
        return nn.SmoothL1Loss()

    elif key == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif key == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    elif key == "focal":
        return FocalLoss(gamma=2.0)

    # -------------------------------------------------------------------------
    # Add new loss functions here. Example:
    # -------------------------------------------------------------------------
    # elif key == "weighted_ce":
    #     return nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))

    else:
        raise ValueError(
            f"Unknown loss function: {key}. "
            f"Available: regression, binary, buckets, mse, mae, l1, huber, "
            f"smooth_l1, cross_entropy, label_smoothing, focal"
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

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model outputs (logits), shape [batch, num_classes]
            targets: Ground truth labels, shape [batch]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
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
