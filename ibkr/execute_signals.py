#!/usr/bin/env python
"""
Execute trade signals from a CSV file via IBKR.

Reads a pending signal file written by a strategy runner, executes each
order through TradeExecutor, and archives the results.

Modes:
    Manual (one-shot):
        python -m ibkr.execute_signals --signals signals/pending_orders.csv

    Watch (poll for new files):
        python -m ibkr.execute_signals --watch --signals-dir signals/

    Dry run (print what would execute, no broker connection):
        python -m ibkr.execute_signals --signals signals/pending_orders.csv --dry-run

Examples:
    # Paper trading, execute a signal file
    python -m ibkr.execute_signals --signals signals/pending_orders.csv

    # Live trading, execute a signal file
    python -m ibkr.execute_signals --signals signals/pending_orders.csv --live

    # Watch mode: poll signals/ dir every 30s, archive after execution
    python -m ibkr.execute_signals --watch --signals-dir signals/ --interval 30

    # Dry run: just show what would happen
    python -m ibkr.execute_signals --signals signals/pending_orders.csv --dry-run
"""

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr.config import IBKRConfig, TradingConfig
from ibkr.trade_executor import TradeExecutor
from ibkr.models import OrderAction

# Use signal_writer's reader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategies'))
from signal_writer import read_signals, SIGNAL_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Location of trading mode lock file
TRADING_MODE_CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_mode.conf')


def check_live_trading_allowed():
    """
    Check trading_mode.conf to see if live trading is enabled.

    Returns True only if TRADING_MODE=live is explicitly set in the config file.
    This is a safety gate — you must manually edit the file to enable live trading.
    """
    if not os.path.exists(TRADING_MODE_CONF):
        return False

    with open(TRADING_MODE_CONF) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            if key.strip() == 'TRADING_MODE' and val.strip().lower() == 'live':
                return True
    return False


def execute_signal_file(filepath, executor, dry_run=False):
    """
    Read and execute all signals in a CSV file.

    Args:
        filepath: Path to signal CSV
        executor: TradeExecutor instance (None if dry_run)
        dry_run: If True, print signals but don't execute

    Returns:
        list of result dicts for the execution log
    """
    signals = read_signals(filepath)
    if not signals:
        logger.info(f"No signals in {filepath}")
        return []

    logger.info(f"Processing {len(signals)} signal(s) from {filepath}")
    results = []

    for sig in signals:
        action = sig['action'].upper()
        symbol = sig['symbol'].upper()
        shares = int(sig.get('shares', 0)) or None  # 0 -> None (auto-size)
        reason = sig.get('reason', '')
        strategy = sig.get('strategy', '')

        label = f"{action} {symbol}"
        if shares:
            label += f" x{shares}"
        if reason:
            label += f" ({reason})"

        if dry_run:
            logger.info(f"[DRY RUN] {label}")
            results.append({
                **sig,
                'status': 'dry_run',
                'fill_price': 0.0,
                'filled_shares': 0,
                'error': '',
                'executed_at': datetime.now().isoformat(timespec='seconds'),
            })
            continue

        logger.info(f"Executing: {label}")

        try:
            if action == 'BUY':
                result = executor.execute_buy(symbol, shares=shares, reason=reason)
            elif action == 'SELL':
                result = executor.execute_sell(symbol, shares=shares, reason=reason)
            else:
                logger.warning(f"Unknown action '{action}' for {symbol}, skipping")
                results.append({
                    **sig,
                    'status': 'skipped',
                    'fill_price': 0.0,
                    'filled_shares': 0,
                    'error': f'unknown action: {action}',
                    'executed_at': datetime.now().isoformat(timespec='seconds'),
                })
                continue

            status = 'filled' if result.success else 'failed'
            fill_price = result.order.avg_fill_price if result.order else 0.0
            filled_shares = result.order.filled_shares if result.order else 0
            error = result.error_message or ''

            logger.info(
                f"  {'OK' if result.success else 'FAIL'}: {symbol} "
                f"{filled_shares} shares @ ${fill_price:.2f}"
                + (f" - {error}" if error else "")
            )

            results.append({
                **sig,
                'status': status,
                'fill_price': fill_price,
                'filled_shares': filled_shares,
                'error': error,
                'executed_at': datetime.now().isoformat(timespec='seconds'),
            })

        except Exception as e:
            logger.error(f"  Exception executing {symbol}: {e}")
            results.append({
                **sig,
                'status': 'error',
                'fill_price': 0.0,
                'filled_shares': 0,
                'error': str(e),
                'executed_at': datetime.now().isoformat(timespec='seconds'),
            })

    return results


def archive_signal_file(filepath, results, archive_dir):
    """
    Move the signal file to the archive directory and write execution results.

    Args:
        filepath: Original signal file path
        results: List of result dicts
        archive_dir: Directory to archive into
    """
    os.makedirs(archive_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.splitext(os.path.basename(filepath))[0]

    # Archive the original signal file
    archive_path = os.path.join(archive_dir, f"{base}_{ts}.csv")
    shutil.move(filepath, archive_path)
    logger.info(f"Archived signal file to {archive_path}")

    # Write execution results alongside
    if results:
        results_path = os.path.join(archive_dir, f"{base}_{ts}_results.csv")
        pd.DataFrame(results).to_csv(results_path, index=False)
        logger.info(f"Execution results written to {results_path}")


def watch_directory(signals_dir, executor, archive_dir, interval_seconds=30,
                    dry_run=False, pattern="pending_orders*.csv"):
    """
    Poll a directory for new signal files and execute them.

    Args:
        signals_dir: Directory to watch
        executor: TradeExecutor instance
        archive_dir: Where to move processed files
        interval_seconds: Seconds between polls
        dry_run: If True, print but don't execute
        pattern: Glob pattern for signal files
    """
    import glob

    logger.info(f"Watching {signals_dir} for signal files (every {interval_seconds}s)...")
    logger.info(f"Pattern: {pattern}")
    logger.info("Press Ctrl+C to stop.\n")

    try:
        while True:
            matches = sorted(glob.glob(os.path.join(signals_dir, pattern)))

            for filepath in matches:
                logger.info(f"\nFound signal file: {filepath}")
                results = execute_signal_file(filepath, executor, dry_run=dry_run)
                archive_signal_file(filepath, results, archive_dir)

            if not matches:
                logger.debug(f"No signal files found, sleeping {interval_seconds}s...")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("\nWatch mode stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Execute trade signals from CSV via IBKR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-shot: execute a signal file (paper trading)
  python -m ibkr.execute_signals --signals signals/pending_orders.csv

  # One-shot: live trading
  python -m ibkr.execute_signals --signals signals/pending_orders.csv --live

  # Watch mode: poll for new signal files
  python -m ibkr.execute_signals --watch --signals-dir signals/ --interval 30

  # Dry run: preview without executing
  python -m ibkr.execute_signals --signals signals/pending_orders.csv --dry-run
        """
    )

    # Signal source
    parser.add_argument("--signals", type=str, default=None,
                        help="Path to signal CSV file (one-shot mode)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: poll for new signal files")
    parser.add_argument("--signals-dir", type=str, default="signals",
                        help="Directory to watch for signal files (default: signals/)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between polls in watch mode (default: 30)")
    parser.add_argument("--archive-dir", type=str, default=None,
                        help="Directory for executed signal archives (default: {signals-dir}/executed/)")

    # Execution mode
    parser.add_argument("--dry-run", action="store_true",
                        help="Print signals without executing (no broker connection)")
    parser.add_argument("--live", action="store_true",
                        help="Use live trading (default: paper)")
    parser.add_argument("--gateway", action="store_true",
                        help="Use IB Gateway instead of TWS")
    parser.add_argument("--client-id", type=int, default=10,
                        help="IBKR client ID (default: 10)")

    # Trading config
    parser.add_argument("--position-size", type=float, default=0.10,
                        help="Position size as fraction of portfolio (default: 0.10)")
    parser.add_argument("--max-positions", type=int, default=10,
                        help="Maximum concurrent positions (default: 10)")
    parser.add_argument("--max-daily-loss", type=float, default=5000,
                        help="Maximum daily loss in dollars (default: 5000)")

    args = parser.parse_args()

    # Validate args
    if not args.signals and not args.watch:
        parser.error("Must specify --signals (one-shot) or --watch (poll mode)")

    archive_dir = args.archive_dir or os.path.join(args.signals_dir, "executed")

    # Dry run: no broker connection needed
    if args.dry_run:
        if args.signals:
            results = execute_signal_file(args.signals, executor=None, dry_run=True)
            if results:
                print(f"\n{len(results)} signal(s) would be executed.")
        elif args.watch:
            watch_directory(args.signals_dir, executor=None, archive_dir=archive_dir,
                            interval_seconds=args.interval, dry_run=True)
        return

    # Configure IBKR connection
    if args.live:
        if not check_live_trading_allowed():
            logger.error("=" * 60)
            logger.error("LIVE TRADING BLOCKED")
            logger.error(f"Edit {TRADING_MODE_CONF} and set TRADING_MODE=live")
            logger.error("=" * 60)
            sys.exit(1)
        ibkr_config = (IBKRConfig.live_gateway(args.client_id) if args.gateway
                        else IBKRConfig.live_tws(args.client_id))
        logger.warning("=" * 60)
        logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
        logger.warning("=" * 60)
    else:
        ibkr_config = (IBKRConfig.paper_gateway(args.client_id) if args.gateway
                        else IBKRConfig.paper_tws(args.client_id))
        logger.info("Paper trading mode")

    trading_config = TradingConfig(
        symbols=set(),  # Not needed — signals specify symbols
        position_size=args.position_size,
        max_positions=args.max_positions,
        max_daily_loss=args.max_daily_loss,
    )

    # Execute
    with TradeExecutor(ibkr_config, trading_config) as executor:
        if args.signals:
            # One-shot mode
            results = execute_signal_file(args.signals, executor)
            archive_signal_file(args.signals, results, archive_dir)
        elif args.watch:
            # Watch mode
            watch_directory(args.signals_dir, executor, archive_dir,
                            interval_seconds=args.interval)


if __name__ == "__main__":
    main()
