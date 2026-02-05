"""
StockFormer: Transformer-based stock prediction pipeline.

Modules:
    - config: Configuration settings and defaults
    - data_utils: Data loading and embedding utilities
    - features: Feature engineering
    - dataset: PyTorch dataset for stock sequences
    - model: StockTransformer model
    - losses: Loss functions
    - training: Training loop utilities
    - main: CLI entry point

Usage:
    # From command line
    python -m stockformer.main --data-path "../../data/all_data_*.csv"

    # From Python
    from stockformer.config import DEFAULT_CONFIG
    from stockformer.main import train, infer

    cfg = DEFAULT_CONFIG.copy()
    cfg["data_path"] = "../../data/all_data_*.csv"
    cfg["horizon"] = 3
    cfg["label_mode"] = "binary"

    train(cfg)
    infer(cfg)
"""

__version__ = "0.1.0"
