"""
Transformer model for stock price prediction.

Classes:
    - PositionalEncoding: Sinusoidal positional encoding for temporal awareness
    - StockTransformer: Transformer encoder for time-series classification/regression
    - GatedCrossAttention: Gated cross-attention layer (stock queries market context)
    - MarketEncoder: Lightweight transformer encoder for market features
    - CrossAttentionStockTransformer: Stock encoder with cross-attention to market context
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
# Gated Cross-Attention (stock ← market context)
# =============================================================================

class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention: stock sequence queries market context.

    The sigmoid gate learns WHEN to use market context:
    - In calm markets, stock-specific signal dominates (gate ≈ 0)
    - In crashes/rallies, market context dominates (gate ≈ 1)

    Architecture:
        attn_out = MultiheadAttention(Q=stock, K=market, V=market)
        gate = sigmoid(W · [stock; attn_out])
        output = LayerNorm(stock + gate * attn_out)

    Reference: MASTER (AAAI 2024), MSGCA pattern
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_seq: torch.Tensor, market_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stock_seq:  [batch, seq_len, d_model] — stock encoder hidden states
            market_seq: [batch, market_len, d_model] — market encoder output

        Returns:
            [batch, seq_len, d_model] — stock states enriched with market context
        """
        attn_out, _ = self.cross_attn(
            query=stock_seq, key=market_seq, value=market_seq,
        )
        attn_out = self.dropout(attn_out)

        # Gating: learn how much market context to inject
        g = self.gate(torch.cat([stock_seq, attn_out], dim=-1))
        return self.norm(stock_seq + g * attn_out)


# =============================================================================
# Market Encoder (lightweight transformer for market features)
# =============================================================================

class MarketEncoder(nn.Module):
    """
    Lightweight transformer encoder for market context features.

    Processes market-level features (VIX, SPY returns, sector ETFs, yields)
    into a context sequence that the stock encoder attends to via cross-attention.

    Smaller than the stock encoder (2 layers vs 4) since market context is
    lower-dimensional and shared across all stocks.
    """

    def __init__(
        self,
        market_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(market_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, market_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_x: [batch, seq_len, market_dim]

        Returns:
            [batch, seq_len, d_model] — encoded market context
        """
        x = self.input_proj(market_x)
        x = self.pos_encoder(x)
        return self.encoder(x)


# =============================================================================
# Cross-Attention Stock Transformer
# =============================================================================

class CrossAttentionStockTransformer(nn.Module):
    """
    Stock transformer with gated cross-attention to market context.

    Architecture:
        1. Market features → MarketEncoder → market context sequence
        2. Stock features → Linear → positional encoding
        3. For each layer:
           a. Self-attention (stock ← stock)
           b. Gated cross-attention (stock ← market)
        4. Attention pooling → output heads

    The gated cross-attention lets the model condition stock predictions
    on the current market regime (bull/bear/sideways).

    Args:
        feature_dim: Number of stock input features per timestep
        market_dim: Number of market context features per timestep
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of stock encoder layers (each gets cross-attention)
        market_layers: Number of market encoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        output_mode: "regression", "binary", or "buckets"
        num_buckets: Number of classes for bucket mode
        layer_drop: Stochastic depth probability
        use_coral: Use CORAL head for ordinal classification
    """

    def __init__(
        self,
        feature_dim: int,
        market_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        market_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_mode: str = "binary",
        num_buckets: int = 5,
        layer_drop: float = 0.0,
        use_coral: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.market_dim = market_dim
        self.d_model = d_model
        self.output_mode = output_mode
        self.num_buckets = num_buckets
        self.use_coral = use_coral

        # Market encoder (lightweight)
        self.market_encoder = MarketEncoder(
            market_dim=market_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=market_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Stock input projection + positional encoding
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Stock self-attention layers + cross-attention after each
        self.stock_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.stock_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True, norm_first=True,
                )
            )
            self.cross_attn_layers.append(
                GatedCrossAttention(d_model=d_model, nhead=nhead, dropout=dropout)
            )

        self.layer_drop = layer_drop
        self.final_norm = nn.LayerNorm(d_model)

        # Attention pooling (same as StockTransformer)
        self.attn_pool = nn.Linear(d_model, 1)

        # Output heads (same architecture as StockTransformer)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 2),
        )
        self.bucket_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_buckets),
        )
        if use_coral and num_buckets > 1:
            self.coral_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model // 2, num_buckets - 1),
            )

    def forward(self, x: torch.Tensor, market_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-attention to market context.

        Args:
            x: Stock features [batch, seq_len, feature_dim]
            market_x: Market features [batch, seq_len, market_dim]

        Returns:
            Predictions, shape depends on output_mode
        """
        # Encode market context
        market_ctx = self.market_encoder(market_x)  # [batch, seq_len, d_model]

        # Project stock features
        stock = self.input_proj(x)  # [batch, seq_len, d_model]
        stock = self.pos_encoder(stock)

        # Alternating self-attention + cross-attention
        for self_attn_layer, cross_attn_layer in zip(self.stock_layers, self.cross_attn_layers):
            # Stochastic depth: skip entire block during training
            if self.training and self.layer_drop > 0 and random.random() < self.layer_drop:
                continue
            stock = self_attn_layer(stock)
            stock = cross_attn_layer(stock, market_ctx)

        stock = self.final_norm(stock)

        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool(stock).squeeze(-1), dim=1)
        pooled = (stock * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Output head
        if self.output_mode == "regression":
            return self.reg_head(pooled).squeeze(-1)
        elif self.output_mode == "binary":
            return self.cls_head(pooled)
        elif self.output_mode == "buckets":
            if self.use_coral:
                return self.coral_head(pooled)
            return self.bucket_head(pooled)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")


# =============================================================================
# Architecture Auto-Detection from Checkpoint
# =============================================================================

def infer_arch_from_state_dict(state_dict, model_path=None):
    """
    Infer model architecture hyperparameters from a saved state_dict.

    This allows inference to automatically match the architecture used during
    training, regardless of what the current config defaults are.

    Args:
        state_dict: Model state dict loaded from a .pt checkpoint
        model_path: Optional path to the .pt file; used to find .meta sidecar

    Returns:
        dict with keys: d_model, num_layers, dim_feedforward, nhead, num_buckets,
        and optionally model_type, market_dim, market_layers for cross-attention
    """
    arch = {}

    # Detect model type: cross-attention models have market_encoder.* keys
    has_market_encoder = any(k.startswith("market_encoder.") for k in state_dict)
    has_cross_attn = any(k.startswith("cross_attn_layers.") for k in state_dict)

    if has_market_encoder and has_cross_attn:
        arch["model_type"] = "cross_attention"

        # market_dim: from market_encoder.input_proj
        if "market_encoder.input_proj.weight" in state_dict:
            arch["market_dim"] = state_dict["market_encoder.input_proj.weight"].shape[1]

        # market_layers: count unique market_encoder.encoder.layers.N
        market_layer_indices = set()
        for key in state_dict:
            if key.startswith("market_encoder.encoder.layers."):
                idx = int(key.split(".")[3])
                market_layer_indices.add(idx)
        if market_layer_indices:
            arch["market_layers"] = max(market_layer_indices) + 1

        # num_layers (stock): count unique stock_layers.N
        stock_layer_indices = set()
        for key in state_dict:
            if key.startswith("stock_layers."):
                idx = int(key.split(".")[1])
                stock_layer_indices.add(idx)
        if stock_layer_indices:
            arch["num_layers"] = max(stock_layer_indices) + 1

        # dim_feedforward: from stock_layers.0.linear1
        if "stock_layers.0.linear1.weight" in state_dict:
            arch["dim_feedforward"] = state_dict["stock_layers.0.linear1.weight"].shape[0]
    else:
        arch["model_type"] = "encoder"

        # num_layers: count unique encoder.layers.N prefixes
        layer_indices = set()
        for key in state_dict:
            if key.startswith("encoder.layers."):
                idx = int(key.split(".")[2])
                layer_indices.add(idx)
        if layer_indices:
            arch["num_layers"] = max(layer_indices) + 1

        # dim_feedforward: from encoder.layers.0.linear1
        if "encoder.layers.0.linear1.weight" in state_dict:
            arch["dim_feedforward"] = state_dict["encoder.layers.0.linear1.weight"].shape[0]

    # d_model: from input_proj (shared by both model types)
    if "input_proj.weight" in state_dict:
        arch["d_model"] = state_dict["input_proj.weight"].shape[0]

    # nhead: read from .meta sidecar if available (exact value saved during training)
    nhead_from_meta = None
    if model_path:
        meta_path = str(model_path) + ".meta"
        try:
            import json as _json
            with open(meta_path) as mf:
                meta = _json.load(mf)
            if "nhead" in meta:
                nhead_from_meta = meta["nhead"]
        except (FileNotFoundError, ValueError, KeyError):
            pass

    if nhead_from_meta is not None:
        arch["nhead"] = nhead_from_meta
    elif "d_model" in arch:
        d = arch["d_model"]
        for candidate in [8, 4, 2, 1]:
            if d % candidate == 0:
                arch["nhead"] = candidate
                break

    # num_buckets: from bucket_head final linear layer
    if "bucket_head.3.weight" in state_dict:
        arch["num_buckets"] = state_dict["bucket_head.3.weight"].shape[0]

    return arch


# =============================================================================
# Model Creation Helper
# =============================================================================

def create_model(feature_dim, label_mode, bucket_edges, cfg, model_type="encoder"):
    """
    Create a StockTransformer model with the appropriate output configuration.

    Args:
        feature_dim: Number of input features
        label_mode: "regression", "binary", or "buckets"
        bucket_edges: List of bucket edges (only used for "buckets" mode)
        cfg: Config dict with model hyperparameters
        model_type: Model architecture type ("encoder" or "cross_attention")

    Returns:
        StockTransformer or CrossAttentionStockTransformer model
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
        num_buckets = cfg.get("_num_buckets_override", len(bucket_edges) + 1)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    # Check if CORAL loss is requested (requires special head)
    use_coral = (cfg.get("loss_name") == "coral" and label_mode == "buckets")

    if model_type == "cross_attention":
        market_dim = cfg.get("market_feature_dim")
        if not market_dim:
            raise ValueError("market_feature_dim must be set in cfg for cross_attention model")

        model = CrossAttentionStockTransformer(
            feature_dim=feature_dim,
            market_dim=market_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            market_layers=cfg.get("market_layers", 2),
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            output_mode=output_mode,
            num_buckets=num_buckets,
            layer_drop=cfg.get("layer_drop", 0.0),
            use_coral=use_coral,
        )
    else:
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
