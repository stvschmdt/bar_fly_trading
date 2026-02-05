# StockFormer

Transformer-based stock return prediction pipeline.

## Quick Start

```bash
# Full run: trains 9 models (3 horizons × 3 label modes) + inference
python -m stockformer.main --data-path "../../data/all_data_*.csv"
```

This creates:
- 9 model checkpoints: `model_{reg,bin,buck}_{3,10,30}d.pt`
- 9 prediction CSVs: `pred_{reg,bin,buck}_{3,10,30}d.csv`
- 9 training logs: `log_{reg,bin,buck}_{3,10,30}d.csv`

## Single Configuration

```bash
# Train + infer with specific settings
python -m stockformer.main --data-path "../../data/all_data_*.csv" \
    --horizon 3 --label-mode binary
```

## Inference Only

```bash
# Run inference with pre-trained model
python -m stockformer.main --data-path "../../data/new_data.csv" \
    --horizon 3 --label-mode binary \
    --model-out trained_model.pt \
    --infer-only
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path` | Path to CSV data (glob patterns supported) | Required |
| `--horizon` | Prediction horizon in days: 3, 10, or 30 | All three |
| `--label-mode` | `regression`, `binary`, or `buckets` | All three |
| `--model-out` | Model checkpoint path | `model.pt` |
| `--output-csv` | Predictions output path | `predictions.csv` |

## Experimentation Options

### Model Architecture
```bash
--d-model 128        # Transformer dimension (default: 128)
--nhead 4            # Attention heads (default: 4)
--num-layers 4       # Transformer layers (default: 4)
--dim-feedforward 256  # FFN dimension (default: 256)
--lookback 20        # Sequence length (default: 20)
```

### Training
```bash
--num-epochs 50      # Training epochs (default: 50)
--batch-size 256     # Batch size (default: 256)
--lr 0.0001          # Learning rate (default: 1e-4)
--dropout 0.1        # Dropout rate (default: 0.1)
--device cuda        # Device: cuda or cpu (auto-detected)
```

### Bucket Classification
```bash
--label-mode buckets --bucket-edges "-6,-4,-2,0,2,4,6"
```

### Embeddings (Correlated Mode)
```bash
--mode correlated \
--market-path market_emb.csv \
--sector-path sector_emb.csv
```

### Inference Options
```bash
--infer-only                    # Skip training
--infer-start-date 2025-01-01   # Filter data from date
--infer-end-date 2025-06-01     # Filter data to date
--output-mode all               # Output: classification, expected_value, probabilities, all
--output-columns core,signals   # Column groups to include (default: all)
```

### Logging & Visualization
```bash
--plot              # Generate training curves plot
--save-config       # Save config JSON for reproducibility
```

## Output Modes

| Mode | Columns |
|------|---------|
| `classification` | pred_class, true_class |
| `expected_value` | pred_class, pred_expected_return, true_class |
| `probabilities` | pred_class, prob_0, prob_1, ..., true_class |
| `all` | All of the above |

## Output Column Groups

Control which input columns appear in prediction CSVs with `--output-columns`:

| Group | Columns |
|-------|---------|
| `core` | date, symbol, open, high, low, adjusted_close, volume |
| `moving_avg` | sma_20, sma_50, sma_200, ema_20, ema_50, ema_200 |
| `technical` | macd, rsi_14, adx_14, atr_14, cci_14, bbands_* |
| `options` | call_volume, put_volume, total_volume, pcr, pcr_14_mean, options_14_* |
| `signals` | macd_signal, adx_signal, rsi_signal, sma_cross_signal, etc. |
| `all` | All 94 input columns (default) |

```bash
# Minimal output (core + predictions)
--output-columns core

# Core + signals + technical
--output-columns core,signals,technical
```

## Smoke Test

```bash
python -m stockformer.smoke_test --data-path "../../data/all_data_*.csv"
```

## Module Structure

```
stockformer/
├── main.py          # CLI entry point
├── config.py        # Configuration defaults
├── data_utils.py    # Data loading and embeddings
├── features.py      # Technical indicators
├── dataset.py       # PyTorch Dataset
├── model.py         # StockTransformer architecture
├── losses.py        # Loss functions
├── training.py      # Training loop
├── inference.py     # Inference and output formatting
├── logging_utils.py # Timing and logging
├── visualization.py # Training curves
└── tests/           # Test suite
```
