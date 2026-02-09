"""
Transformer model for stock price prediction.

Classes:
    - PositionalEncoding: Sinusoidal positional encoding for temporal awareness
    - StockTransformer: Transformer encoder for time-series classification/regression
"""

import copy
import math
import random

import torch
import torch.nn as nn


# =============================================================================
# Positional Encoding
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# =============================================================================
# Stochastic Depth Encoder (LayerDrop)
# =============================================================================

class StochasticTransformerEncoder(nn.Module):
    """
    Transformer encoder with stochastic depth (LayerDrop).

    Randomly skips layers during training with probability drop_prob.
    All layers are used during evaluation.
    """

    def __init__(self, encoder_layer, num_layers, drop_prob=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.drop_prob = drop_prob
        self.norm = nn.LayerNorm(encoder_layer.self_attn.in_proj_weight.size(1))

    def forward(self, x, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            if self.training and random.random() < self.drop_prob:
                continue
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


# =============================================================================
# Model
# =============================================================================

class StockTransformer(nn.Module):
    """
    Transformer encoder for stock time-series classification/regression.

    Architecture:
        1. Linear projection: feature_dim -> d_model
        2. Positional encoding: adds temporal position information
        3. Transformer encoder: processes the sequence
        4. Attention pooling: weighted aggregation over sequence
        5. MLP output head: produces predictions based on output_mode

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
        layer_drop: float = 0.0,
        use_coral: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model
        self.output_mode = output_mode
        self.num_buckets = num_buckets
        self.use_coral = use_coral

        # Project input features to model dimension
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder (with optional stochastic depth)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        if layer_drop > 0:
            self.encoder = StochasticTransformerEncoder(
                encoder_layer, num_layers=num_layers, drop_prob=layer_drop
            )
        else:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # MLP output heads (2-layer with GELU + dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.bucket_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_buckets),
        )
        # CORAL head: K-1 logits for ordinal classification
        if use_coral and num_buckets > 1:
            self.coral_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_buckets - 1),
            )

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

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        enc = self.encoder(x)  # [batch, seq_len, d_model]

        # Attention pooling over sequence
        attn_weights = torch.softmax(self.attn_pool(enc).squeeze(-1), dim=1)  # [batch, seq_len]
        pooled = (enc * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch, d_model]

        # Apply appropriate output head
        if self.output_mode == "regression":
            return self.reg_head(pooled).squeeze(-1)  # [batch]

        elif self.output_mode == "binary":
            return self.cls_head(pooled)  # [batch, 2]

        elif self.output_mode == "buckets":
            if self.use_coral:
                return self.coral_head(pooled)  # [batch, num_buckets - 1]
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

    # Check if CORAL loss is requested (requires special head)
    use_coral = (cfg.get("loss_name") == "coral" and label_mode == "buckets")

    model = StockTransformer(
        feature_dim=feature_dim,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        output_mode=output_mode,
        num_buckets=num_buckets,
        layer_drop=cfg.get("layer_drop", 0.0),
        use_coral=use_coral,
    )

    return model
