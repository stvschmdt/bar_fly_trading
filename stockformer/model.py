"""
Transformer model for stock price prediction.

Classes:
    - StockTransformer: Original transformer encoder (bidirectional)
    - CrossAttentionStockTransformer: Decoder with causal masking + cross-attention to market context
"""

import torch
import torch.nn as nn
import math


# =============================================================================
# Utility Functions
# =============================================================================

def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate a causal mask for decoder self-attention.

    Returns a mask where position i can only attend to positions <= i.
    True values are masked out (cannot attend).

    Args:
        seq_len: Length of the sequence
        device: Device to create tensor on

    Returns:
        Boolean mask of shape [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


# =============================================================================
# Market Encoder
# =============================================================================

class MarketEncoder(nn.Module):
    """
    Encoder for market context (SPY, QQQ, sector ETFs).

    Processes market-level time series and produces context embeddings
    that individual stocks can attend to via cross-attention.

    Architecture:
        1. Linear projection: market_feature_dim -> d_model
        2. Positional encoding
        3. Transformer encoder (bidirectional - market context is fully observable)

    Args:
        market_feature_dim: Number of input features per timestep for market data
        d_model: Model dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of encoder layers (default: 2)
        dim_feedforward: Hidden dimension of FFN (default: 256)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length for positional encoding (default: 512)

    Input shape: [batch, seq_len, market_feature_dim]
    Output shape: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        market_feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model

        # Project market features to model dimension
        self.input_proj = nn.Linear(market_feature_dim, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)

        # Transformer encoder (bidirectional for market context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode market context.

        Args:
            x: Market features [batch, seq_len, market_feature_dim]

        Returns:
            Market context embeddings [batch, seq_len, d_model]
        """
        seq_len = x.size(1)

        # Project to model dimension
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Encode (bidirectional attention)
        encoded = self.encoder(x)  # [batch, seq_len, d_model]

        return encoded


# =============================================================================
# Cross-Attention Stock Transformer (Decoder with Market Context)
# =============================================================================

class CrossAttentionStockTransformer(nn.Module):
    """
    Stock prediction model with causal self-attention and cross-attention to market context.

    Architecture:
        1. Market Encoder: Encodes SPY/QQQ/sector features
        2. Stock Input Projection: stock_feature_dim -> d_model
        3. Decoder layers with:
           - Causal self-attention (can only see past stock data)
           - Cross-attention to market context
        4. Output head for predictions

    This is similar to encoder-decoder transformers used in translation,
    but adapted for time-series forecasting with market context.

    Args:
        stock_feature_dim: Number of input features per timestep for individual stock
        market_feature_dim: Number of input features per timestep for market data
        d_model: Model dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_encoder_layers: Number of market encoder layers (default: 2)
        num_decoder_layers: Number of stock decoder layers (default: 3)
        dim_feedforward: Hidden dimension of FFN (default: 256)
        dropout: Dropout probability (default: 0.1)
        output_mode: "regression", "binary", or "buckets" (default: "binary")
        num_buckets: Number of output classes for buckets mode (default: 5)
        max_seq_len: Maximum sequence length (default: 512)

    Input:
        stock_x: [batch, seq_len, stock_feature_dim]
        market_x: [batch, seq_len, market_feature_dim]

    Output shape depends on output_mode:
        - regression: [batch]
        - binary: [batch, 2]
        - buckets: [batch, num_buckets]
    """

    def __init__(
        self,
        stock_feature_dim: int,
        market_feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_mode: str = "binary",
        num_buckets: int = 5,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.stock_feature_dim = stock_feature_dim
        self.market_feature_dim = market_feature_dim
        self.d_model = d_model
        self.output_mode = output_mode
        self.num_buckets = num_buckets

        # Market encoder
        self.market_encoder = MarketEncoder(
            market_feature_dim=market_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Stock input projection
        self.stock_proj = nn.Linear(stock_feature_dim, d_model)

        # Positional encoding for stock decoder
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)

        # Decoder layers (with causal self-attention + cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, 2)
        self.bucket_head = nn.Linear(d_model, num_buckets)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self,
        stock_x: torch.Tensor,
        market_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with market context.

        Args:
            stock_x: Stock features [batch, seq_len, stock_feature_dim]
            market_x: Market features [batch, seq_len, market_feature_dim]

        Returns:
            Predictions, shape depends on output_mode
        """
        batch_size, seq_len, _ = stock_x.shape
        device = stock_x.device

        # Encode market context (bidirectional)
        market_context = self.market_encoder(market_x)  # [batch, seq_len, d_model]

        # Project stock features
        stock_emb = self.stock_proj(stock_x)  # [batch, seq_len, d_model]

        # Add positional encoding
        stock_emb = stock_emb + self.pos_encoding[:, :seq_len, :]
        stock_emb = self.dropout(stock_emb)

        # Generate causal mask for decoder self-attention
        causal_mask = generate_causal_mask(seq_len, device)

        # Decode with causal self-attention + cross-attention to market
        decoded = self.decoder(
            tgt=stock_emb,
            memory=market_context,
            tgt_mask=causal_mask,
        )  # [batch, seq_len, d_model]

        # Pool by taking the last timestep
        pooled = decoded[:, -1, :]  # [batch, d_model]

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
# Original Model (kept for backward compatibility)
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

def create_model(feature_dim, label_mode, bucket_edges, cfg, model_type="encoder"):
    """
    Create a StockTransformer model with the appropriate output configuration.

    Args:
        feature_dim: Number of input features (for stock or combined features)
        label_mode: "regression", "binary", or "buckets"
        bucket_edges: List of bucket edges (only used for "buckets" mode)
        cfg: Config dict with model hyperparameters
        model_type: "encoder" for original bidirectional, "cross_attention" for new architecture

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
        num_buckets = len(bucket_edges) + 1
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    if model_type == "encoder":
        # Original bidirectional encoder model
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
    elif model_type == "cross_attention":
        # New cross-attention model with market context
        # feature_dim should be stock features only
        # market_feature_dim comes from cfg
        market_feature_dim = cfg.get("market_feature_dim", feature_dim)
        model = CrossAttentionStockTransformer(
            stock_feature_dim=feature_dim,
            market_feature_dim=market_feature_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_encoder_layers=cfg.get("num_encoder_layers", 2),
            num_decoder_layers=cfg.get("num_decoder_layers", cfg["num_layers"]),
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            output_mode=output_mode,
            num_buckets=num_buckets,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'encoder' or 'cross_attention'")

    return model
