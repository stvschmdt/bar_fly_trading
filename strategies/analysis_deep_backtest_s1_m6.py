#!/usr/bin/env python
"""
Deep backtest analysis for S1 (Oversold Bounce) and M6 (52-Week Low Bounce).

Vectorized analysis using forward returns from CSV data.
NOT a backtest runner — validates strategy conditions before building.

Usage:
    python strategies/analysis_deep_backtest_s1_m6.py --data-path 'all_data_*.csv'
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def load_data(data_path):
    """Load CSV data with glob support."""
    if '*' in data_path:
        files = sorted(glob(data_path))
        if not files:
            print(f"No files matching {data_path}")
            sys.exit(1)
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(data_path)

    # Normalize column names
    if 'ticker' in df.columns and 'symbol' not in df.columns:
        df = df.rename(columns={'ticker': 'symbol'})
    if 'close' in df.columns and 'adjusted_close' not in df.columns:
        df = df.rename(columns={'close': 'adjusted_close'})

    df['date'] = pd.to_datetime(df['date'])
    return df


def strategy_stats(name, mask, returns, df):
    """Compute detailed stats for a strategy."""
    rets = returns.dropna()
    if len(rets) == 0:
        print(f"\n{'='*70}")
        print(f"  {name}: NO SIGNALS")
        print(f"{'='*70}")
        return

    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    win_rate = wins / len(rets) * 100

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Signals:     {len(rets):,}")
    print(f"  Win Rate:    {win_rate:.1f}% ({wins} wins, {losses} losses)")
    print(f"  Avg Return:  {rets.mean():.2f}%")
    print(f"  Median Ret:  {rets.median():.2f}%")
    print(f"  Std Dev:     {rets.std():.2f}%")
    sharpe = rets.mean() / rets.std() if rets.std() > 0 else 0
    print(f"  Sharpe:      {sharpe:.3f}")
    print()

    # Percentiles
    pcts = [5, 10, 25, 50, 75, 90, 95]
    print(f"  Return Distribution:")
    for p in pcts:
        print(f"    {p:>3}th pct: {np.percentile(rets, p):+.2f}%")
    print()

    # Monthly breakdown
    signal_dates = df.loc[mask, 'date'].dropna()
    signal_rets = pd.DataFrame({'date': signal_dates.values[:len(rets)], 'return': rets.values[:len(signal_dates)]})
    signal_rets['month'] = pd.to_datetime(signal_rets['date']).dt.to_period('M')
    monthly = signal_rets.groupby('month')['return'].agg(['count', 'mean', lambda x: (x > 0).mean() * 100])
    monthly.columns = ['trades', 'avg_return', 'win_rate']
    print(f"  Monthly Breakdown:")
    print(f"  {'Month':<12} {'Trades':>8} {'Avg Ret':>10} {'Win Rate':>10}")
    print(f"  {'-'*42}")
    for period, row in monthly.iterrows():
        print(f"  {str(period):<12} {row['trades']:>8.0f} {row['avg_return']:>+9.2f}% {row['win_rate']:>9.1f}%")
    print()

    # Top 10 best/worst
    signal_df = df.loc[mask, ['date', 'symbol']].copy()
    signal_df = signal_df.iloc[:len(rets)]
    signal_df['return'] = rets.values[:len(signal_df)]

    print(f"  Top 10 Best Trades:")
    best = signal_df.nlargest(10, 'return')
    for _, row in best.iterrows():
        d = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        print(f"    {row['symbol']:<8} {d} {row['return']:>+8.2f}%")

    print(f"\n  Top 10 Worst Trades:")
    worst = signal_df.nsmallest(10, 'return')
    for _, row in worst.iterrows():
        d = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        print(f"    {row['symbol']:<8} {d} {row['return']:>+8.2f}%")

    # Consecutive losses (max drawdown sequence)
    is_loss = (rets <= 0).values
    max_streak = 0
    current_streak = 0
    for loss in is_loss:
        if loss:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    print(f"\n  Max Consecutive Losses: {max_streak}")

    # Sector breakdown (if available)
    if 'sector' in df.columns:
        signal_df['sector'] = df.loc[mask, 'sector'].values[:len(signal_df)]
        sector_stats = signal_df.groupby('sector')['return'].agg(['count', 'mean', lambda x: (x > 0).mean() * 100])
        sector_stats.columns = ['trades', 'avg_return', 'win_rate']
        sector_stats = sector_stats.sort_values('trades', ascending=False)
        print(f"\n  Sector Breakdown:")
        print(f"  {'Sector':<25} {'Trades':>8} {'Avg Ret':>10} {'Win Rate':>10}")
        print(f"  {'-'*55}")
        for sector, row in sector_stats.head(15).iterrows():
            print(f"  {str(sector):<25} {row['trades']:>8.0f} {row['avg_return']:>+9.2f}% {row['win_rate']:>9.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="Deep backtest analysis for S1 and M6")
    parser.add_argument("--data-path", type=str, default="all_data_*.csv",
                        help="Path to CSV data (supports globs)")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.data_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Check required columns
    required = ['rsi_14', 'bollinger_bands_signal', 'adjusted_close', 'bbands_lower_20',
                '52_week_low', 'bull_bear_delta', 'future_3_day_pct', 'future_30_day_pct']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nWARNING: Missing columns: {missing}")
        print(f"Available columns: {sorted(df.columns.tolist())}")

    # =========================================================================
    # S1: Oversold Bounce (3-day hold)
    # =========================================================================
    if all(c in df.columns for c in ['rsi_14', 'bollinger_bands_signal', 'adjusted_close', 'bbands_lower_20', 'future_3_day_pct']):
        s1_mask = (
            (df['rsi_14'] < 35) &
            (df['bollinger_bands_signal'] == 1) &
            (df['adjusted_close'] < df['bbands_lower_20'])
        )
        s1_returns = df.loc[s1_mask, 'future_3_day_pct'] * 100  # decimal → percentage
        strategy_stats("S1: Oversold Bounce (3-day hold)", s1_mask, s1_returns, df)
    else:
        print("\nS1: Missing required columns, skipping")

    # =========================================================================
    # M6: 52-Week Low Bounce (30-day hold)
    # =========================================================================
    if all(c in df.columns for c in ['rsi_14', 'bull_bear_delta', 'adjusted_close', '52_week_low', 'future_30_day_pct']):
        m6_mask = (
            (df['adjusted_close'] < df['52_week_low'] * 1.10) &
            (df['rsi_14'] < 40) &
            (df['bull_bear_delta'] <= 0)
        )
        m6_returns = df.loc[m6_mask, 'future_30_day_pct'] * 100  # decimal → percentage
        strategy_stats("M6: 52-Week Low Bounce (30-day hold)", m6_mask, m6_returns, df)
    else:
        print("\nM6: Missing required columns, skipping")


if __name__ == "__main__":
    main()
