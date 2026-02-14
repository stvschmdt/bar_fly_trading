"""
Main entry point for the StockFormer pipeline.

Usage:
    # Run all horizons and label modes (9 combinations)
    python -m stockformer.main --data-path "../../data/all_data_*.csv"

    # Run single configuration (train + infer)
    python -m stockformer.main --data-path "../../data/all_data_*.csv" --horizon 3 --label-mode binary

    # Inference only with pre-trained model
    python -m stockformer.main --data-path "../../data/new_data.csv" --horizon 3 --label-mode binary \
        --model-out trained_model.pt --infer-only --output-mode all

    # Inference with date filtering
    python -m stockformer.main --data-path "../../data/all_data_*.csv" --horizon 3 --label-mode binary \
        --model-out trained_model.pt --infer-only --infer-start-date 2025-01-01
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_CONFIG, MARKET_FEATURE_COLUMNS, get_feature_columns, get_target_column
from .data_utils import (
    load_panel_csvs,
    add_future_returns,
    load_embeddings,
    merge_embeddings,
)
from .features import add_all_features
from .sector_features import extract_etf_features, add_sector_features, exclude_etf_symbols
from .dataset import StockSequenceDataset, make_train_val_split, compute_quantile_edges
from .model import create_model
from .losses import get_loss_function
from .training import get_optimizer, train_model
from .logging_utils import (
    Timer,
    get_gpu_memory,
    reset_gpu_memory_stats,
    log_data_summary,
    print_data_summary,
    log_model_summary,
    print_model_summary,
    log_training_summary,
    print_training_summary,
    save_run_config,
)
from .visualization import plot_training_curves, plot_training_curves_from_log
from .inference import infer, filter_data_by_date  # Import inference from dedicated module


# =============================================================================
# Train Function
# =============================================================================

def train(cfg):
    """
    Train a StockTransformer model.

    Args:
        cfg: Configuration dict (see config.py for structure)

    Returns:
        Dict with training history
    """
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    # Start overall timer
    with Timer("Total training") as total_timer:

        # Load data
        print("\nLoading main panel data...")
        with Timer("Data loading") as load_timer:
            df = load_panel_csvs(cfg["data_path"])

            # Filter training data by date range if specified
            train_start_date = cfg.get("train_start_date")
            train_end_date = cfg.get("train_end_date")
            if train_start_date or train_end_date:
                print(f"Filtering training data: {train_start_date or 'start'} to {train_end_date or 'end'}...")
                df = filter_data_by_date(df, start_date=train_start_date, end_date=train_end_date)

            # Add future returns
            print("Adding future returns...")
            df = add_future_returns(df, horizons=[cfg["horizon"]])

            # Add technical features
            print("Adding technical features...")
            df = add_all_features(df)

            # Add sector/market ETF features (real data, replaces dummy embeddings)
            print("Extracting sector/market ETF features...")
            etf_features = extract_etf_features(df)
            df = add_sector_features(df, etf_features)
            df = exclude_etf_symbols(df)

            # Load embeddings (auto-create if missing)
            print("Loading embeddings...")
            market_result = None
            sector_result = None

            if cfg["market_path"]:
                market_result = load_embeddings(cfg["market_path"], prefix="m_", base_df=df)

            if cfg["sector_path"]:
                sector_result = load_embeddings(cfg["sector_path"], prefix="s_", base_df=df)

            # Merge embeddings if in correlated mode
            if cfg["mode"] == "correlated":
                df = merge_embeddings(df, market_result, sector_result)

        print(f"Data loading time: {load_timer}")

        # Get feature columns based on model type
        model_type = cfg.get("model_type", "encoder")
        target_col = get_target_column(cfg["horizon"])

        if model_type == "cross_attention":
            # Cross-attention: stock features exclude market cols, market passed separately
            feature_cols = get_feature_columns(df, cfg["mode"])
            market_feature_cols = [c for c in MARKET_FEATURE_COLUMNS if c in df.columns]
            # Remove market features from stock features to avoid duplication
            feature_cols = [c for c in feature_cols if c not in market_feature_cols]
            cfg["market_feature_dim"] = len(market_feature_cols)
            print(f"\nCross-attention: {len(feature_cols)} stock features, {len(market_feature_cols)} market features")
        else:
            feature_cols = get_feature_columns(df, cfg["mode"])
            market_feature_cols = None

        # Log data summary
        data_summary = log_data_summary(df, feature_cols, target_col)
        print_data_summary(data_summary)

        # Auto-compute quantile bucket edges if requested
        if cfg["label_mode"] == "buckets" and cfg.get("bucket_edges") == "auto":
            n_buckets = cfg.get("n_buckets", 4)
            cfg["bucket_edges"] = compute_quantile_edges(df, target_col, n_buckets=n_buckets)
            print(f"Auto bucket edges ({n_buckets} buckets): {cfg['bucket_edges']}")

        print(f"\nFeatures: {len(feature_cols)} columns")
        print(f"Target: {target_col}")

        # Create dataset
        dataset = StockSequenceDataset(
            df=df,
            lookback=cfg["lookback"],
            target_col=target_col,
            feature_cols=feature_cols,
            label_mode=cfg["label_mode"],
            bucket_edges=cfg["bucket_edges"],
            market_feature_cols=market_feature_cols,
            binary_threshold=cfg.get("binary_threshold", 0.0),
            min_return_threshold=cfg.get("min_return_threshold", 0.0),
        )

        # Train/val split
        train_ds, val_ds = make_train_val_split(dataset, val_fraction=cfg["val_fraction"])
        print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

        # Create data loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
        )

        # Setup device
        device = cfg["device"]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Reset GPU memory stats if using CUDA
        if device == "cuda":
            reset_gpu_memory_stats()

        # Create model
        model = create_model(
            feature_dim=len(feature_cols),
            label_mode=cfg["label_mode"],
            bucket_edges=cfg["bucket_edges"],
            cfg=cfg,
            model_type=model_type,
        ).to(device)

        # Log model summary
        model_summary = log_model_summary(model)
        print_model_summary(model_summary)

        # Get loss function and optimizer
        # Allow regression-compatible losses through; block classification-only losses
        _regression_losses = {"logcosh", "directional_mse", "combined_regression", "mse", "mae", "l1", "huber", "smooth_l1"}
        _requested = cfg.get("loss_name")
        if cfg["label_mode"] == "regression" and _requested and _requested not in _regression_losses:
            effective_loss_name = None  # skip classification losses for regression
        else:
            effective_loss_name = _requested
        loss_fn = get_loss_function(
            cfg["label_mode"], loss_name=effective_loss_name,
            direction_weight=cfg.get("direction_weight", 3.0),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
        optimizer = get_optimizer(model, cfg["optimizer"], cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 0.01))

        loss_desc = type(loss_fn).__name__
        if effective_loss_name:
            loss_desc += f" (override: {effective_loss_name})"
        elif cfg.get("loss_name") and cfg["label_mode"] == "regression":
            loss_desc += f" (skipped {cfg['loss_name']} — regression uses MSE)"
        if cfg.get("entropy_reg_weight", 0) > 0 and cfg["label_mode"] != "regression":
            loss_desc += f" + entropy_reg={cfg['entropy_reg_weight']}"
        print(f"Loss: {loss_desc}")

        # Train
        print("\nStarting training...")
        with Timer("Model training") as train_timer:
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                label_mode=cfg["label_mode"],
                num_epochs=cfg["num_epochs"],
                model_out_path=cfg["model_out"],
                log_path=cfg["log_path"],
                patience=cfg.get("patience", 10),
                warmup_epochs=cfg.get("warmup_epochs", 5),
                entropy_reg_weight=cfg.get("entropy_reg_weight", 0.0),
                model_type=model_type,
                collapse_lr_reduction=cfg.get("collapse_lr_reduction", 0.0),
                collapse_entropy_boost=cfg.get("collapse_entropy_boost", 0.0),
                checkpoint_every=cfg.get("checkpoint_every", 5),
            )

    # Get GPU memory usage
    gpu_memory = get_gpu_memory() if device == "cuda" else None

    # Log and print training summary
    training_summary = log_training_summary(
        history=history,
        train_time=total_timer.elapsed,
        device=device,
        gpu_memory=gpu_memory,
    )
    print_training_summary(training_summary)

    # Generate training curves if plot flag is set
    if cfg.get("plot") and cfg.get("log_path"):
        plot_path = cfg["log_path"].replace(".csv", "_curves.png")
        plot_training_curves(history, output_path=plot_path)

    # Save run config if specified
    if cfg.get("save_config") and cfg.get("log_path"):
        config_path = cfg["log_path"].replace(".csv", "_config.json")
        save_run_config(cfg, config_path)

    return history


# Note: infer() function is now in inference.py and imported above


# =============================================================================
# Run All Horizons
# =============================================================================

def run_all_horizons(cfg):
    """
    Run train + infer for all combinations of horizons and label modes.

    Horizons: 3, 10, 30 days
    Label modes: regression, binary, buckets

    Total: 9 runs
    """
    targets = [
        (3, "3d"),
        (10, "10d"),
        (30, "30d"),
    ]
    label_modes = [
        ("regression", "reg"),
        ("binary", "bin"),
        ("buckets", "buck"),
    ]

    # Parse base paths
    base_model = cfg["model_out"]
    base_log = cfg["log_path"]
    base_pred = cfg["output_csv"]

    def split_ext(path, default_ext):
        if path is None:
            return None, ""
        base, ext = os.path.splitext(path)
        if not ext:
            ext = default_ext
        return base, ext

    model_base, model_ext = split_ext(base_model, ".pt")
    log_base, log_ext = split_ext(base_log, ".csv")
    pred_base, pred_ext = split_ext(base_pred, ".csv")

    for label_mode, label_tag in label_modes:
        for horizon, tag in targets:
            print("\n" + "=" * 60)
            print(f"RUN: horizon={horizon}, label_mode={label_mode}")
            print("=" * 60)

            suffix = f"{label_tag}_{tag}"

            # Create config for this run
            run_cfg = cfg.copy()
            run_cfg["horizon"] = horizon
            run_cfg["label_mode"] = label_mode

            # Set output paths with suffix
            run_cfg["model_out"] = (
                f"{model_base}_{suffix}{model_ext}"
                if model_base is not None
                else f"model_{suffix}.pt"
            )
            run_cfg["log_path"] = (
                f"{log_base}_{suffix}{log_ext}" if log_base is not None else None
            )
            run_cfg["output_csv"] = (
                f"{pred_base}_{suffix}{pred_ext}"
                if pred_base is not None
                else f"predictions_{suffix}.csv"
            )

            # Train and infer
            train(run_cfg)
            infer(run_cfg)


# =============================================================================
# CLI Argument Parser
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="StockFormer: Transformer-based stock prediction pipeline"
    )

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_CONFIG["data_path"],
        help="Path or glob pattern for panel CSVs",
    )
    parser.add_argument(
        "--market-path",
        type=str,
        default=DEFAULT_CONFIG["market_path"],
        help="Path to market embeddings CSV",
    )
    parser.add_argument(
        "--sector-path",
        type=str,
        default=DEFAULT_CONFIG["sector_path"],
        help="Path to sector embeddings CSV",
    )

    # Sequence / label settings
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_CONFIG["lookback"],
        help="Number of past days in each sequence",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction horizon (3, 10, 30). If set with --label-mode, runs single config",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default=None,
        choices=["regression", "binary", "buckets"],
        help="Label mode. If set with --horizon, runs single config",
    )
    parser.add_argument(
        "--bucket-edges",
        type=str,
        default=None,
        help="Comma-separated bucket edges for 'buckets' mode, e.g. '-6,-4,-2,0,2,4,6'",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=None,
        help="Number of buckets when using --bucket-edges auto (default: 4)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_CONFIG["mode"],
        choices=["single", "correlated"],
        help="Feature mode: single (stock only) or correlated (with embeddings)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_CONFIG["val_fraction"],
        help="Validation fraction",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULT_CONFIG["optimizer"],
        choices=["adam", "adamw", "sgd", "rmsprop"],
        help="Optimizer",
    )

    # Anti-collapse / loss settings
    parser.add_argument(
        "--loss-name",
        type=str,
        default=DEFAULT_CONFIG["loss_name"],
        choices=[
            "focal", "label_smoothing", "cross_entropy", "ordinal_focal",
            "logcosh", "directional_mse", "combined_regression",
            "symmetric_ce", "soft_ordinal", "coral",
            None,
        ],
        help="Override loss function (default: standard CE for classification, MSE for regression)",
    )
    parser.add_argument(
        "--entropy-reg-weight",
        type=float,
        default=DEFAULT_CONFIG["entropy_reg_weight"],
        help="Entropy regularization weight to prevent prediction collapse (0 = disabled)",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=DEFAULT_CONFIG["binary_threshold"],
        help="Binary label threshold: class 1 = return >= this value (default: 0.005 = +0.5%%)",
    )
    parser.add_argument(
        "--min-return-threshold",
        type=float,
        default=DEFAULT_CONFIG["min_return_threshold"],
        help="Filter samples with |return| < this from classification training (default: 0.0025)",
    )
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=DEFAULT_CONFIG["direction_weight"],
        help="DirectionalMSE penalty for wrong-sign predictions (default: 3.0)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_CONFIG["weight_decay"],
        help="AdamW weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["patience"],
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_CONFIG["warmup_epochs"],
        help="Linear warmup epochs (default: 5)",
    )

    # Model architecture
    parser.add_argument(
        "--d-model",
        type=int,
        default=DEFAULT_CONFIG["d_model"],
        help="Transformer model dimension",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=DEFAULT_CONFIG["nhead"],
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_CONFIG["num_layers"],
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=DEFAULT_CONFIG["dim_feedforward"],
        help="Feedforward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_CONFIG["dropout"],
        help="Dropout probability",
    )
    parser.add_argument(
        "--layer-drop",
        type=float,
        default=DEFAULT_CONFIG["layer_drop"],
        help="Stochastic depth drop probability (0 = disabled, default: 0.1)",
    )

    # Cross-attention model
    parser.add_argument(
        "--model-type",
        type=str,
        default="encoder",
        choices=["encoder", "cross_attention"],
        help="Model architecture: encoder (stock-only) or cross_attention (stock + market context)",
    )
    parser.add_argument(
        "--market-layers",
        type=int,
        default=2,
        help="Number of transformer layers in market encoder (cross_attention only, default: 2)",
    )

    # Research loss tuning
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (default: 2.0, research: 1.5)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for focal/CE loss (default: 0, research: 0.05)",
    )

    # Collapse recovery
    parser.add_argument(
        "--collapse-lr-reduction",
        type=float,
        default=0.0,
        help="On collapse, multiply LR by this factor (0 = halt, research: 0.1)",
    )
    parser.add_argument(
        "--collapse-entropy-boost",
        type=float,
        default=0.0,
        help="On collapse, multiply entropy_reg by this factor (0 = no boost, research: 2.0)",
    )

    # System
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_CONFIG["num_workers"],
        help="DataLoader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG["device"],
        help="Device (cuda, cpu, or auto)",
    )

    # Output paths
    parser.add_argument(
        "--model-out",
        type=str,
        default=DEFAULT_CONFIG["model_out"],
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=DEFAULT_CONFIG["log_path"],
        help="Training log CSV path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_CONFIG["output_csv"],
        help="Predictions output CSV path",
    )

    # Logging and visualization
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training curves plot after training",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save run configuration to JSON for reproducibility",
    )

    # Training date filter
    parser.add_argument(
        "--train-start-date",
        type=str,
        default=None,
        help="Start date for training data (YYYY-MM-DD). Data before this is excluded from training.",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=None,
        help="End date for training data (YYYY-MM-DD). Data after this is excluded from training.",
    )

    # Inference options
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="Run inference only (skip training). Requires pre-trained model at --model-out",
    )
    parser.add_argument(
        "--infer-start-date",
        type=str,
        default=None,
        help="Start date for inference data filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--infer-end-date",
        type=str,
        default=None,
        help="End date for inference data filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="all",
        choices=["classification", "expected_value", "probabilities", "all"],
        help="Inference output format: classification (class only), expected_value (weighted return), "
             "probabilities (full prob vector), or all (default)",
    )
    parser.add_argument(
        "--output-columns",
        type=str,
        default="all",
        help="Comma-separated column groups to include in output: "
             "core,moving_avg,technical,options,signals,all (default: all)",
    )

    return parser.parse_args()


def args_to_config(args):
    """Convert parsed args to config dict."""
    cfg = DEFAULT_CONFIG.copy()

    cfg["data_path"] = args.data_path
    cfg["market_path"] = args.market_path
    cfg["sector_path"] = args.sector_path
    cfg["lookback"] = args.lookback
    cfg["mode"] = args.mode
    cfg["batch_size"] = args.batch_size
    cfg["lr"] = args.lr
    cfg["num_epochs"] = args.epochs
    cfg["val_fraction"] = args.val_fraction
    cfg["optimizer"] = args.optimizer
    cfg["loss_name"] = args.loss_name
    cfg["entropy_reg_weight"] = args.entropy_reg_weight
    cfg["binary_threshold"] = args.binary_threshold
    cfg["min_return_threshold"] = args.min_return_threshold
    cfg["direction_weight"] = args.direction_weight
    cfg["weight_decay"] = args.weight_decay
    cfg["patience"] = args.patience
    cfg["warmup_epochs"] = args.warmup_epochs
    cfg["d_model"] = args.d_model
    cfg["nhead"] = args.nhead
    cfg["num_layers"] = args.num_layers
    cfg["dim_feedforward"] = args.dim_feedforward
    cfg["dropout"] = args.dropout
    cfg["layer_drop"] = args.layer_drop
    cfg["model_type"] = args.model_type
    cfg["market_layers"] = args.market_layers
    cfg["focal_gamma"] = args.focal_gamma
    cfg["label_smoothing"] = args.label_smoothing
    cfg["collapse_lr_reduction"] = args.collapse_lr_reduction
    cfg["collapse_entropy_boost"] = args.collapse_entropy_boost
    cfg["num_workers"] = args.num_workers
    cfg["device"] = args.device
    cfg["model_out"] = args.model_out
    cfg["log_path"] = args.log_path
    cfg["output_csv"] = args.output_csv

    # Logging and visualization flags
    cfg["plot"] = args.plot
    cfg["save_config"] = args.save_config

    # Training date filter
    cfg["train_start_date"] = args.train_start_date
    cfg["train_end_date"] = args.train_end_date

    # Inference options
    cfg["infer_only"] = args.infer_only
    cfg["infer_start_date"] = args.infer_start_date
    cfg["infer_end_date"] = args.infer_end_date
    cfg["output_mode"] = args.output_mode
    cfg["output_columns"] = args.output_columns

    # Parse bucket edges ("auto" for quantile-based, or comma-separated floats)
    if args.bucket_edges:
        if args.bucket_edges.strip().lower() == "auto":
            cfg["bucket_edges"] = "auto"
        else:
            cfg["bucket_edges"] = [float(x.strip()) for x in args.bucket_edges.split(",")]
    if args.n_buckets is not None:
        cfg["n_buckets"] = args.n_buckets

    # Set horizon and label_mode if provided
    if args.horizon is not None:
        cfg["horizon"] = args.horizon
    if args.label_mode is not None:
        cfg["label_mode"] = args.label_mode

    return cfg


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    cfg = args_to_config(args)

    # Inference-only mode
    if args.infer_only:
        if args.horizon is None or args.label_mode is None:
            raise ValueError("--infer-only requires both --horizon and --label-mode")
        if not os.path.exists(cfg["model_out"]):
            raise FileNotFoundError(f"Model not found: {cfg['model_out']}. "
                                    "Use --model-out to specify a trained model.")
        print(f"Running inference only: horizon={args.horizon}, label_mode={args.label_mode}")
        print(f"Model: {cfg['model_out']}")
        print(f"Output mode: {cfg['output_mode']}")
        infer(cfg)

    # Single config: train + infer
    elif args.horizon is not None and args.label_mode is not None:
        print(f"Running single config: horizon={args.horizon}, label_mode={args.label_mode}")
        train(cfg)
        infer(cfg)

    # All 9 combinations
    else:
        print("Running all horizon/label_mode combinations...")
        run_all_horizons(cfg)
