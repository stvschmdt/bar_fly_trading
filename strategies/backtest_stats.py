"""
Backtest statistics, trade log reporting, and symbol list I/O.

Computes per-symbol P&L, win/loss metrics, and Sharpe ratio from trade logs.
Writes trade logs and symbol lists to CSV for downstream consumption.
"""

import math

import pandas as pd


def compute_stats(trades, start_cash):
    """
    Compute backtest statistics from a list of completed trades.

    Args:
        trades: List of trade dicts with keys:
            symbol, entry_date, exit_date, entry_price, exit_price,
            shares, pnl, return_pct, hold_days
        start_cash: Initial portfolio value

    Returns:
        Dict with overall and per-symbol stats
    """
    if not trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_return_pct': 0.0,
            'avg_hold_days': 0.0,
            'sharpe_ratio': 0.0,
            'per_symbol': {},
        }

    df = pd.DataFrame(trades)

    # Overall stats
    wins = int((df['pnl'] > 0).sum())
    losses = int((df['pnl'] <= 0).sum())
    total_pnl = df['pnl'].sum()
    avg_hold = df['hold_days'].mean()

    # Sharpe from trade returns (annualized)
    returns = df['return_pct'] / 100.0
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = 252.0 / max(avg_hold, 1)
        sharpe = (returns.mean() / returns.std()) * math.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Per-symbol stats
    per_symbol = {}
    for symbol, group in df.groupby('symbol'):
        sym_wins = int((group['pnl'] > 0).sum())
        sym_losses = int((group['pnl'] <= 0).sum())
        sym_total = len(group)
        per_symbol[symbol] = {
            'trades': sym_total,
            'wins': sym_wins,
            'losses': sym_losses,
            'win_rate': sym_wins / sym_total * 100 if sym_total > 0 else 0,
            'total_pnl': group['pnl'].sum(),
            'avg_return_pct': group['return_pct'].mean(),
            'avg_hold_days': group['hold_days'].mean(),
        }

    return {
        'total_trades': len(df),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(df) * 100,
        'total_pnl': total_pnl,
        'avg_return_pct': df['return_pct'].mean(),
        'avg_hold_days': avg_hold,
        'sharpe_ratio': sharpe,
        'per_symbol': per_symbol,
    }


def print_stats(stats, start_cash, final_value):
    """Print formatted backtest statistics."""
    print("\n" + "=" * 70)
    print("TRADE STATISTICS")
    print("=" * 70)

    total_return = (final_value - start_cash) / start_cash * 100

    print(f"""
Portfolio:
  Starting:      ${start_cash:,.2f}
  Ending:        ${final_value:,.2f}
  Total Return:  {total_return:+.2f}%

Trade Summary:
  Total Trades:  {stats['total_trades']}
  Wins:          {stats['wins']}
  Losses:        {stats['losses']}
  Win Rate:      {stats['win_rate']:.1f}%
  Total P&L:     ${stats['total_pnl']:+,.2f}
  Avg Return:    {stats['avg_return_pct']:+.2f}%
  Avg Hold:      {stats['avg_hold_days']:.1f} days
  Sharpe Ratio:  {stats['sharpe_ratio']:.2f}
""")

    # Per-symbol breakdown
    if stats['per_symbol']:
        print("-" * 70)
        print(f"{'Symbol':<8} {'Trades':>6} {'Wins':>5} {'Losses':>6} "
              f"{'Win%':>6} {'P&L':>12} {'Avg Ret%':>9} {'Avg Hold':>9}")
        print("-" * 70)

        sorted_symbols = sorted(
            stats['per_symbol'].items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True,
        )

        for symbol, s in sorted_symbols:
            print(f"{symbol:<8} {s['trades']:>6} {s['wins']:>5} {s['losses']:>6} "
                  f"{s['win_rate']:>5.1f}% ${s['total_pnl']:>+10,.2f} "
                  f"{s['avg_return_pct']:>+8.2f}% {s['avg_hold_days']:>8.1f}d")

        print("-" * 70)

    print("=" * 70)


def write_trade_log(trades, filepath):
    """Write trade log to CSV."""
    if not trades:
        print(f"No trades to write.")
        return

    df = pd.DataFrame(trades)
    cols = ['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price',
            'shares', 'pnl', 'return_pct', 'hold_days']
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(filepath, index=False)
    print(f"Trade log written to: {filepath} ({len(df)} trades)")


def write_symbols(symbols, filepath):
    """Write symbol list to CSV (single 'symbol' column)."""
    sorted_syms = sorted(symbols)
    df = pd.DataFrame({'symbol': sorted_syms})
    df.to_csv(filepath, index=False)
    print(f"Symbol list written to: {filepath} ({len(sorted_syms)} symbols)")


def write_rankings(per_symbol_stats, filepath):
    """Write per-symbol backtest rankings to CSV.

    Columns: symbol, trades, wins, losses, win_rate, total_pnl,
             avg_return_pct, avg_hold_days, score

    The 'score' column is win_rate * avg_return_pct — rewards both
    consistency and magnitude (higher = better).
    """
    if not per_symbol_stats:
        print("No per-symbol stats to write.")
        return

    rows = []
    for symbol, s in sorted(per_symbol_stats.items()):
        rows.append({
            'symbol': symbol,
            'trades': s['trades'],
            'wins': s['wins'],
            'losses': s['losses'],
            'win_rate': round(s['win_rate'], 2),
            'total_pnl': round(s['total_pnl'], 2),
            'avg_return_pct': round(s['avg_return_pct'], 4),
            'avg_hold_days': round(s['avg_hold_days'], 1),
            'score': round(s['win_rate'] * s['avg_return_pct'], 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Rankings written to: {filepath} ({len(df)} symbols)")


def read_symbols(filepath):
    """Read symbol list from CSV or text file.

    Supports:
        - CSV with 'symbol' or 'ticker' column header
        - Plain text with one symbol per line (no header)
    """
    df = pd.read_csv(filepath)
    if 'symbol' in df.columns:
        return list(df['symbol'].dropna().unique())
    elif 'ticker' in df.columns:
        return list(df['ticker'].dropna().unique())
    else:
        # Assume first column is symbols (no header or unknown header)
        return list(df.iloc[:, 0].dropna().unique())
