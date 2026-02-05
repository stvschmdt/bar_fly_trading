"""
Trading Strategy Analysis using ML Predictions

Analyzes all 9 prediction files and proposes trading strategies
based on prediction fields and signal fields.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load prediction files
pred_dir = Path("/home/stvschmdt/proj/mlr/stockformer/output/predictions")

# Load binary predictions (most useful for entry/exit signals)
bin_3d = pd.read_csv(pred_dir / "pred_bin_3d.csv")
bin_10d = pd.read_csv(pred_dir / "pred_bin_10d.csv")
bin_30d = pd.read_csv(pred_dir / "pred_bin_30d.csv")

# Load regression predictions
reg_3d = pd.read_csv(pred_dir / "pred_reg_3d.csv")
reg_10d = pd.read_csv(pred_dir / "pred_reg_10d.csv")
reg_30d = pd.read_csv(pred_dir / "pred_reg_30d.csv")

# Load buckets predictions
buck_3d = pd.read_csv(pred_dir / "pred_buck_3d.csv")
buck_10d = pd.read_csv(pred_dir / "pred_buck_10d.csv")

# Compute weighted expected return from bucket probabilities
# Bucket edges: -2%, 0%, 2% → midpoints: -3%, -1%, +1%, +3%
BUCKET_MIDPOINTS = [-0.03, -0.01, 0.01, 0.03]

def compute_bucket_expected_return(df, prefix=""):
    """Compute expected return as probability-weighted bucket midpoints."""
    prob_cols = [f'prob_{i}' for i in range(4)]
    if all(col in df.columns for col in prob_cols):
        expected = sum(df[f'prob_{i}'] * BUCKET_MIDPOINTS[i] for i in range(4))
        return expected
    return None

buck_3d['buck_exp_ret_3d'] = compute_bucket_expected_return(buck_3d)
buck_10d['buck_exp_ret_10d'] = compute_bucket_expected_return(buck_10d)

print(f"Loaded {len(bin_3d)} rows from binary 3d predictions")
print(f"Date range: {bin_3d['date'].min()} to {bin_3d['date'].max()}")
print(f"Tickers: {bin_3d['ticker'].nunique()}")

# Signal columns available
signal_cols = ['macd_signal', 'macd_zero_signal', 'adx_signal', 'atr_signal',
               'pe_ratio_signal', 'bollinger_bands_signal', 'rsi_signal',
               'sma_cross_signal', 'cci_signal', 'pcr_signal']

# Prediction columns
pred_cols_bin = ['pred_class', 'pred_expected_return', 'prob_0', 'prob_1']
pred_cols_reg = ['pred_return']

print(f"\nSignal columns: {signal_cols}")
print(f"Binary prediction columns: {pred_cols_bin}")


def simulate_strategy(df, entry_condition, exit_condition, hold_min=2, hold_max=13, strategy_name="Strategy"):
    """
    Simulate a trading strategy with entry/exit conditions.

    Returns trades with entry date, exit date, return, and hold time.
    """
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    trades = []

    # Group by ticker
    for ticker, group in df.groupby('ticker'):
        group = group.reset_index(drop=True)
        i = 0

        while i < len(group) - hold_min:
            row = group.iloc[i]

            # Check entry condition
            if entry_condition(row):
                entry_date = row['date']
                entry_price = row['close']

                # Look for exit within hold_max days
                exited = False
                for j in range(hold_min, min(hold_max + 1, len(group) - i)):
                    exit_row = group.iloc[i + j]

                    if exit_condition(exit_row) or j == hold_max:
                        exit_date = exit_row['date']
                        exit_price = exit_row['close']
                        hold_days = j

                        pct_return = (exit_price - entry_price) / entry_price * 100

                        trades.append({
                            'ticker': ticker,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'hold_days': hold_days,
                            'pct_return': pct_return,
                            'win': pct_return > 0
                        })

                        i += j  # Skip to after exit
                        exited = True
                        break

                if not exited:
                    i += 1
            else:
                i += 1

    return pd.DataFrame(trades)


def analyze_strategy(trades_df, strategy_name, df):
    """Analyze strategy performance."""
    if len(trades_df) == 0:
        return None

    # Calculate metrics
    total_trades = len(trades_df)
    win_rate = trades_df['win'].mean() * 100
    avg_return = trades_df['pct_return'].mean()
    total_return = trades_df['pct_return'].sum()
    avg_hold = trades_df['hold_days'].mean()

    # Calculate trades per month
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['month'] = trades_df['entry_date'].dt.to_period('M')
    trades_per_month = trades_df.groupby('month').size().mean()

    # Separate wins and losses
    wins = trades_df[trades_df['win']]
    losses = trades_df[~trades_df['win']]

    avg_win = wins['pct_return'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pct_return'].mean() if len(losses) > 0 else 0

    return {
        'strategy': strategy_name,
        'total_trades': total_trades,
        'trades_per_month': round(trades_per_month, 1),
        'win_rate': round(win_rate, 1),
        'avg_return': round(avg_return, 2),
        'total_return': round(total_return, 1),
        'avg_hold_days': round(avg_hold, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2)
    }


# Join binary predictions from different horizons
print("\n" + "="*70)
print("PREPARING DATA: Joining predictions from multiple horizons")
print("="*70)

# Merge binary predictions on date and ticker
merged = bin_3d[['date', 'ticker', 'close', 'pred_class', 'pred_expected_return', 'prob_0', 'prob_1'] + signal_cols].copy()
merged.columns = ['date', 'ticker', 'close', 'pred_class_3d', 'pred_exp_ret_3d', 'prob0_3d', 'prob1_3d'] + signal_cols

# Add 10d predictions
bin_10d_sub = bin_10d[['date', 'ticker', 'pred_class', 'pred_expected_return', 'prob_0', 'prob_1']].copy()
bin_10d_sub.columns = ['date', 'ticker', 'pred_class_10d', 'pred_exp_ret_10d', 'prob0_10d', 'prob1_10d']
merged = merged.merge(bin_10d_sub, on=['date', 'ticker'], how='inner')

# Add 30d predictions
bin_30d_sub = bin_30d[['date', 'ticker', 'pred_class', 'pred_expected_return', 'prob_0', 'prob_1']].copy()
bin_30d_sub.columns = ['date', 'ticker', 'pred_class_30d', 'pred_exp_ret_30d', 'prob0_30d', 'prob1_30d']
merged = merged.merge(bin_30d_sub, on=['date', 'ticker'], how='inner')

# Add regression predictions
reg_3d_sub = reg_3d[['date', 'ticker', 'pred_return']].copy()
reg_3d_sub.columns = ['date', 'ticker', 'pred_reg_3d']
merged = merged.merge(reg_3d_sub, on=['date', 'ticker'], how='inner')

reg_10d_sub = reg_10d[['date', 'ticker', 'pred_return']].copy()
reg_10d_sub.columns = ['date', 'ticker', 'pred_reg_10d']
merged = merged.merge(reg_10d_sub, on=['date', 'ticker'], how='inner')

# Add bucket-derived expected returns (probability-weighted midpoints)
buck_3d_sub = buck_3d[['date', 'ticker', 'buck_exp_ret_3d']].copy()
merged = merged.merge(buck_3d_sub, on=['date', 'ticker'], how='inner')

buck_10d_sub = buck_10d[['date', 'ticker', 'buck_exp_ret_10d']].copy()
merged = merged.merge(buck_10d_sub, on=['date', 'ticker'], how='inner')

# Add future returns for validation (from original data)
merged_full = bin_3d[['date', 'ticker', 'future_3_day_pct', 'future_10_day_pct', 'future_30_day_pct']].copy()
merged = merged.merge(merged_full, on=['date', 'ticker'], how='inner')

print(f"Merged dataset: {len(merged)} rows")
print(f"Columns: {list(merged.columns)}")


# Define strategies
print("\n" + "="*70)
print("STRATEGY ANALYSIS")
print("="*70)

strategies = []

# Strategy 1: Multi-horizon Bullish Confirmation
# Entry: All 3 horizons predict positive (class 1) AND rsi_signal not overbought
# Exit: 3d prediction flips to negative OR hold_max reached
def strat1_entry(row):
    return (row['pred_class_3d'] == 1 and
            row['pred_class_10d'] == 1 and
            row['pred_class_30d'] == 1 and
            row['rsi_signal'] != 1)  # Not overbought

def strat1_exit(row):
    return row['pred_class_3d'] == 0

print("\n--- Strategy 1: Multi-Horizon Bullish Confirmation ---")
print("Entry: pred_class_3d=1 AND pred_class_10d=1 AND pred_class_30d=1 AND rsi_signal!=1")
print("Exit: pred_class_3d=0 OR max hold reached")
trades1 = simulate_strategy(merged, strat1_entry, strat1_exit, hold_min=2, hold_max=13, strategy_name="Multi-Horizon Bullish")
result1 = analyze_strategy(trades1, "Multi-Horizon Bullish", merged)
if result1:
    print(f"Results: {result1}")
    strategies.append(result1)


# Strategy 2: High Confidence + Technical Signal
# Entry: prob1_3d > 0.6 AND macd_signal > 0 AND sma_cross_signal > 0
# Exit: prob1_3d < 0.45 OR macd_signal < 0
def strat2_entry(row):
    return (row['prob1_3d'] > 0.6 and
            row['macd_signal'] > 0 and
            row['sma_cross_signal'] > 0)

def strat2_exit(row):
    return row['prob1_3d'] < 0.45 or row['macd_signal'] < 0

print("\n--- Strategy 2: High Confidence + Technical ---")
print("Entry: prob1_3d>0.6 AND macd_signal>0 AND sma_cross_signal>0")
print("Exit: prob1_3d<0.45 OR macd_signal<0")
trades2 = simulate_strategy(merged, strat2_entry, strat2_exit, hold_min=2, hold_max=13, strategy_name="High Confidence Technical")
result2 = analyze_strategy(trades2, "High Confidence Technical", merged)
if result2:
    print(f"Results: {result2}")
    strategies.append(result2)


# Strategy 3: Regression + Momentum
# Entry: pred_reg_3d > 0.01 AND pred_reg_10d > 0.02 AND adx_signal > 0
# Exit: pred_reg_3d < 0 OR cci_signal < 0
def strat3_entry(row):
    return (row['pred_reg_3d'] > 0.01 and
            row['pred_reg_10d'] > 0.02 and
            row['adx_signal'] > 0)

def strat3_exit(row):
    return row['pred_reg_3d'] < 0 or row['cci_signal'] < 0

print("\n--- Strategy 3: Regression + Momentum ---")
print("Entry: pred_reg_3d>0.01 AND pred_reg_10d>0.02 AND adx_signal>0")
print("Exit: pred_reg_3d<0 OR cci_signal<0")
trades3 = simulate_strategy(merged, strat3_entry, strat3_exit, hold_min=2, hold_max=13, strategy_name="Regression Momentum")
result3 = analyze_strategy(trades3, "Regression Momentum", merged)
if result3:
    print(f"Results: {result3}")
    strategies.append(result3)


# Strategy 4: Conservative Multi-Signal
# Entry: pred_class_3d=1 AND bollinger_bands_signal=1 AND pcr_signal>0
# Exit: pred_class_3d=0 AND rsi_signal=1 (overbought)
def strat4_entry(row):
    return (row['pred_class_3d'] == 1 and
            row['bollinger_bands_signal'] == 1 and
            row['pcr_signal'] > 0)

def strat4_exit(row):
    return row['pred_class_3d'] == 0 or row['rsi_signal'] == 1

print("\n--- Strategy 4: Conservative Multi-Signal ---")
print("Entry: pred_class_3d=1 AND bollinger_bands_signal=1 AND pcr_signal>0")
print("Exit: pred_class_3d=0 OR rsi_signal=1")
trades4 = simulate_strategy(merged, strat4_entry, strat4_exit, hold_min=2, hold_max=13, strategy_name="Conservative Multi-Signal")
result4 = analyze_strategy(trades4, "Conservative Multi-Signal", merged)
if result4:
    print(f"Results: {result4}")
    strategies.append(result4)


# Strategy 5: Expected Return Threshold
# Entry: pred_exp_ret_3d > 0.002 AND pred_exp_ret_10d > 0.005 AND macd_zero_signal > 0
# Exit: pred_exp_ret_3d < 0 OR atr_signal < 0
def strat5_entry(row):
    return (row['pred_exp_ret_3d'] > 0.002 and
            row['pred_exp_ret_10d'] > 0.005 and
            row['macd_zero_signal'] > 0)

def strat5_exit(row):
    return row['pred_exp_ret_3d'] < 0 or row['atr_signal'] < 0

print("\n--- Strategy 5: Expected Return Threshold ---")
print("Entry: pred_exp_ret_3d>0.002 AND pred_exp_ret_10d>0.005 AND macd_zero_signal>0")
print("Exit: pred_exp_ret_3d<0 OR atr_signal<0")
trades5 = simulate_strategy(merged, strat5_entry, strat5_exit, hold_min=2, hold_max=13, strategy_name="Expected Return Threshold")
result5 = analyze_strategy(trades5, "Expected Return Threshold", merged)
if result5:
    print(f"Results: {result5}")
    strategies.append(result5)


# Strategy 6: Probability Spread
# Entry: prob1_3d - prob0_3d > 0.2 AND cci_signal > 0
# Exit: prob1_3d < prob0_3d OR hold max
def strat6_entry(row):
    return ((row['prob1_3d'] - row['prob0_3d']) > 0.2 and
            row['cci_signal'] > 0)

def strat6_exit(row):
    return row['prob1_3d'] < row['prob0_3d']

print("\n--- Strategy 6: Probability Spread ---")
print("Entry: (prob1_3d - prob0_3d) > 0.2 AND cci_signal>0")
print("Exit: prob1_3d < prob0_3d")
trades6 = simulate_strategy(merged, strat6_entry, strat6_exit, hold_min=2, hold_max=13, strategy_name="Probability Spread")
result6 = analyze_strategy(trades6, "Probability Spread", merged)
if result6:
    print(f"Results: {result6}")
    strategies.append(result6)


# Strategy 7: Bucket Expected Return (probability-weighted)
# Entry: buck_exp_ret_3d > 0.005 AND buck_exp_ret_10d > 0.01 AND macd_signal > 0
# Exit: buck_exp_ret_3d < 0 OR adx_signal < 0
def strat7_entry(row):
    return (row['buck_exp_ret_3d'] > 0.005 and
            row['buck_exp_ret_10d'] > 0.01 and
            row['macd_signal'] > 0)

def strat7_exit(row):
    return row['buck_exp_ret_3d'] < 0 or row['adx_signal'] < 0

print("\n--- Strategy 7: Bucket Expected Return ---")
print("Entry: buck_exp_ret_3d>0.5% AND buck_exp_ret_10d>1% AND macd_signal>0")
print("Exit: buck_exp_ret_3d<0 OR adx_signal<0")
trades7 = simulate_strategy(merged, strat7_entry, strat7_exit, hold_min=2, hold_max=13, strategy_name="Bucket Expected Return")
result7 = analyze_strategy(trades7, "Bucket Expected Return", merged)
if result7:
    print(f"Results: {result7}")
    strategies.append(result7)


# Strategy 8: Dual Regressor Agreement (regression + bucket expected return)
# Entry: pred_reg_3d > 0.01 AND buck_exp_ret_3d > 0.005 AND rsi_signal != 1
# Exit: pred_reg_3d < 0 AND buck_exp_ret_3d < 0
def strat8_entry(row):
    return (row['pred_reg_3d'] > 0.01 and
            row['buck_exp_ret_3d'] > 0.005 and
            row['rsi_signal'] != 1)

def strat8_exit(row):
    return row['pred_reg_3d'] < 0 and row['buck_exp_ret_3d'] < 0

print("\n--- Strategy 8: Dual Regressor Agreement ---")
print("Entry: pred_reg_3d>1% AND buck_exp_ret_3d>0.5% AND rsi_signal!=1 (not overbought)")
print("Exit: pred_reg_3d<0 AND buck_exp_ret_3d<0 (both bearish)")
trades8 = simulate_strategy(merged, strat8_entry, strat8_exit, hold_min=2, hold_max=13, strategy_name="Dual Regressor Agreement")
result8 = analyze_strategy(trades8, "Dual Regressor Agreement", merged)
if result8:
    print(f"Results: {result8}")
    strategies.append(result8)


# Strategy 9: Strong Bucket Signal + Binary Confirmation
# Entry: buck_exp_ret_3d > 0.01 AND pred_class_3d == 1 AND sma_cross_signal > 0
# Exit: buck_exp_ret_3d < 0.005 OR pred_class_3d == 0
def strat9_entry(row):
    return (row['buck_exp_ret_3d'] > 0.01 and
            row['pred_class_3d'] == 1 and
            row['sma_cross_signal'] > 0)

def strat9_exit(row):
    return row['buck_exp_ret_3d'] < 0.005 or row['pred_class_3d'] == 0

print("\n--- Strategy 9: Strong Bucket + Binary Confirmation ---")
print("Entry: buck_exp_ret_3d>1% AND pred_class_3d=1 AND sma_cross_signal>0")
print("Exit: buck_exp_ret_3d<0.5% OR pred_class_3d=0")
trades9 = simulate_strategy(merged, strat9_entry, strat9_exit, hold_min=2, hold_max=13, strategy_name="Strong Bucket + Binary")
result9 = analyze_strategy(trades9, "Strong Bucket + Binary", merged)
if result9:
    print(f"Results: {result9}")
    strategies.append(result9)


# Filter strategies with >= 3 trades per month
valid_strategies = [s for s in strategies if s['trades_per_month'] >= 3]

# Sort by win rate then by avg return
valid_strategies.sort(key=lambda x: (x['win_rate'], x['avg_return']), reverse=True)

# ============================================================================
# FINAL SYNOPSIS
# ============================================================================

def print_banner(text, char="═", width=74):
    print(f"\n{char*width}")
    print(f"  {text}")
    print(f"{char*width}")

def print_table_row(cols, widths):
    row = "│"
    for col, w in zip(cols, widths):
        row += f" {str(col):<{w}} │"
    print(row)

def print_table_sep(widths, char="─"):
    sep = "├"
    for i, w in enumerate(widths):
        sep += char * (w + 2)
        sep += "┼" if i < len(widths) - 1 else "┤"
    print(sep)

def print_table_top(widths):
    sep = "┌"
    for i, w in enumerate(widths):
        sep += "─" * (w + 2)
        sep += "┬" if i < len(widths) - 1 else "┐"
    print(sep)

def print_table_bottom(widths):
    sep = "└"
    for i, w in enumerate(widths):
        sep += "─" * (w + 2)
        sep += "┴" if i < len(widths) - 1 else "┘"
    print(sep)

print_banner("STOCKFORMER STRATEGY ANALYSIS RESULTS", "═")

print(f"""
  📊 Data Summary
  ─────────────────────────────────────────────────────────────────────
  • Inference Period:  2024-07-01 to 2025-11-11
  • Total Rows:        {len(merged):,}
  • Unique Tickers:    {merged['ticker'].nunique()}
  • Models Used:       Binary (3d/10d/30d) + Regression (3d/10d) + Buckets (3d/10d)
  • Signal Fields:     {len(signal_cols)} technical indicators
""")

print_banner("TOP STRATEGIES (min 3 trades/month, hold 2-13 days)", "─")

if valid_strategies:
    # Table header
    widths = [3, 26, 8, 9, 8, 9, 6]
    headers = ["#", "Strategy", "Win %", "Trades/Mo", "Avg Ret", "Avg Hold", "Total"]

    print()
    print_table_top(widths)
    print_table_row(headers, widths)
    print_table_sep(widths, "═")

    for i, s in enumerate(valid_strategies[:5], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        cols = [
            f"{medal}",
            s['strategy'][:26],
            f"{s['win_rate']}%",
            f"{s['trades_per_month']}",
            f"+{s['avg_return']}%",
            f"{s['avg_hold_days']}d",
            f"{s['total_return']:.0f}%"
        ]
        print_table_row(cols, widths)
        if i < len(valid_strategies[:5]):
            print_table_sep(widths)

    print_table_bottom(widths)

    # Detailed breakdown of top 3
    print_banner("DETAILED BREAKDOWN - TOP 3", "─")

    for i, strat in enumerate(valid_strategies[:3], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"

        # Win/loss bar
        wins = int(strat['win_rate'] / 5)
        losses = 20 - wins
        bar = "█" * wins + "░" * losses

        print(f"""
  {medal} #{i} {strat['strategy']}
  ───────────────────────────────────────────────────────────────────

  Win Rate:      [{bar}] {strat['win_rate']}%

  ┌─────────────────┬─────────────────┬─────────────────┐
  │ Total Trades    │ Trades/Month    │ Avg Hold Days   │
  │ {strat['total_trades']:<15} │ {strat['trades_per_month']:<15} │ {strat['avg_hold_days']:<15} │
  ├─────────────────┼─────────────────┼─────────────────┤
  │ Avg Win         │ Avg Loss        │ Avg Return      │
  │ +{strat['avg_win']:<14}% │ {strat['avg_loss']:<15}% │ +{strat['avg_return']:<14}% │
  └─────────────────┴─────────────────┴─────────────────┘

  Total Return: {strat['total_return']:.1f}% (sum of all trades)
""")

else:
    print("\n  ⚠️  No strategies met the minimum 3 trades/month requirement")
    print("\n  All strategies tested:")
    for s in strategies:
        print(f"    • {s['strategy']}: {s['trades_per_month']} trades/month, {s['win_rate']}% win rate")

# Summary recommendation
print_banner("RECOMMENDATION", "═")

if valid_strategies:
    best = valid_strategies[0]
    print(f"""
  ✅ Best Strategy: {best['strategy']}

  • Highest win rate at {best['win_rate']}% with {best['trades_per_month']} trades/month
  • Average hold time of {best['avg_hold_days']} days fits the 2-13 day constraint
  • Risk/reward: +{best['avg_win']}% avg win vs {best['avg_loss']}% avg loss

  Entry signals used: ML predictions + technical indicators
  Exit signals used:  ML prediction reversal + technical confirmation
""")

print("═" * 74)
print("  Analysis complete. Output saved to console.")
print("═" * 74 + "\n")