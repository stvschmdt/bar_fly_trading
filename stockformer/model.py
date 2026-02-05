"""
Transformer model for stock price prediction.

Classes:
    - StockTransformer: Transformer encoder for time-series classification/regression
"""

import torch
import torch.nn as nn


# =============================================================================
# Model
# =============================================================================

class StockTransformer(nn.Module):
    """
    Simple transformer encoder for stock time-series classification/regression.

    Architecture:
        1. Linear projection: feature_dim -> d_model
        2. Transformer encoder: processes the sequence
        3. Pooling: takes the last timestep
        4. Output head: produces predictions based on output_mode

    Args:
        feature_dim: Number of input features per timestep
        d_model: Transformer model dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer encoder layers (default: 3)
        dim_feedforward: Hidden dimension of feedforward network (default: 256)
        dropout: Dropout probability (default: 0.1)
        output_mode: Type of output
            - "regression": single scalar output
            - "binary": 2-class classification logits
            - "buckets": multi-class classification logits
        num_buckets: Number of output classes for "buckets" mode (default: 5)

    Input shape: [batch, seq_len, feature_dim]
    Output shape:
        - regression: [batch]
        - binary: [batch, 2]
        - buckets: [batch, num_buckets]
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_mode: str = "binary",
        num_buckets: int = 5,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model
        self.output_mode = output_mode
        self.num_buckets = num_buckets

        # Project input features to model dimension
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads for different modes
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, 2)
        self.bucket_head = nn.Linear(d_model, num_buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, feature_dim]

        Returns:
            Output tensor, shape depends on output_mode
        """
        # Project to model dimension
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Pass through transformer encoder
        enc = self.encoder(x)  # [batch, seq_len, d_model]

        # Pool by taking the last timestep
        pooled = enc[:, -1, :]  # [batch, d_model]

        # Apply appropriate output head
        if self.output_mode == "regression":
            return self.reg_head(pooled).squeeze(-1)  # [batch]

        elif self.output_mode == "binary":
            return self.cls_head(pooled)  # [batch, 2]

        elif self.output_mode == "buckets":
            return self.bucket_head(pooled)  # [batch, num_buckets]

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")


# =============================================================================
# Model Creation Helper
# =============================================================================

def create_model(feature_dim, label_mode, bucket_edges, cfg):
    """
    Create a StockTransformer model with the appropriate output configuration.

    Args:
        feature_dim: Number of input features
        label_mode: "regression", "binary", or "buckets"
        bucket_edges: List of bucket edges (only used for "buckets" mode)
        cfg: Config dict with model hyperparameters

    Returns:
        StockTransformer model
    """
    # Determine output mode and number of buckets
    if label_mode == "regression":
        output_mode = "regression"
        num_buckets = 1
    elif label_mode == "binary":
        output_mode = "binary"
        num_buckets = 2
    elif label_mode == "buckets":
        if not bucket_edges:
            raise ValueError("bucket_edges must be provided for 'buckets' label_mode.")
        output_mode = "buckets"
        num_buckets = len(bucket_edges) + 1
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    model = StockTransformer(
        feature_dim=feature_dim,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        output_mode=output_mode,
        num_buckets=num_buckets,
    )

    return model
