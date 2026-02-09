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
from zoneinfo import ZoneInfo

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

# Daily rejection de-dup: only notify once per (symbol, reason_type) per day.
# Prevents spamming when RT loop runs every 15 minutes and a symbol stays
# in the same rejected state all day.
_rejections_notified: dict[str, set[tuple[str, str]]] = {}  # {date_str: set((symbol, reason))}


def _is_new_rejection(symbol: str, reason_type: str) -> bool:
    """Check if this rejection is new today (not yet notified)."""
    today = datetime.now().strftime('%Y-%m-%d')
    # Reset on new day
    if today not in _rejections_notified:
        _rejections_notified.clear()
        _rejections_notified[today] = set()
    key = (symbol.upper(), reason_type)
    if key in _rejections_notified[today]:
        return False
    _rejections_notified[today].add(key)
    return True


def is_market_open() -> bool:
    """Check if US stock market is currently open (9:30-16:00 ET, Mon-Fri)."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    # Weekend
    if now_et.weekday() >= 5:
        return False
    # Before open or after close
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et < market_close


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


def log_account_state(executor, label=""):
    """Log verbose account and portfolio state."""
    prefix = f"[{label}] " if label else ""
    logger.info("=" * 70)
    logger.info(f"{prefix}ACCOUNT & PORTFOLIO STATE")
    logger.info("=" * 70)

    account = executor.get_account_summary()
    if account:
        logger.info(f"  Net Liquidation:    ${account.net_liquidation:>14,.2f}")
        logger.info(f"  Total Cash:         ${account.total_cash:>14,.2f}")
        logger.info(f"  Available Funds:    ${account.available_funds:>14,.2f}")
        logger.info(f"  Buying Power:       ${account.buying_power:>14,.2f}")
        logger.info(f"  Gross Positions:    ${account.gross_position_value:>14,.2f}")
        logger.info(f"  Unrealized P&L:     ${account.unrealized_pnl:>14,.2f}")
        logger.info(f"  Realized P&L:       ${account.realized_pnl:>14,.2f}")
    else:
        logger.warning(f"  Could not retrieve account summary")

    positions = executor.get_position_summary()
    open_positions = positions.get('positions', {})
    if open_positions:
        logger.info(f"  Open Positions ({len(open_positions)}):")
        for sym, pos in open_positions.items():
            shares = getattr(pos, 'shares', pos.get('shares', 0) if isinstance(pos, dict) else 0)
            avg_cost = getattr(pos, 'avg_cost', pos.get('avg_cost', 0) if isinstance(pos, dict) else 0)
            mkt_val = getattr(pos, 'market_value', pos.get('market_value', 0) if isinstance(pos, dict) else 0)
            logger.info(f"    {sym:6s}  {shares:>4} shares  avg_cost=${avg_cost:>8.2f}  mkt_val=${mkt_val:>10,.2f}")
    else:
        logger.info(f"  Open Positions: none")
    logger.info("=" * 70)
    return account


def format_account_state_text(executor, label=""):
    """Format account state as text for email body."""
    prefix = f"[{label}] " if label else ""
    lines = []
    lines.append(f"{prefix}ACCOUNT & PORTFOLIO STATE")
    lines.append("=" * 50)

    account = executor.get_account_summary()
    if account:
        lines.append(f"  Net Liquidation:    ${account.net_liquidation:>14,.2f}")
        lines.append(f"  Total Cash:         ${account.total_cash:>14,.2f}")
        lines.append(f"  Available Funds:    ${account.available_funds:>14,.2f}")
        lines.append(f"  Buying Power:       ${account.buying_power:>14,.2f}")
        lines.append(f"  Gross Positions:    ${account.gross_position_value:>14,.2f}")
        lines.append(f"  Unrealized P&L:     ${account.unrealized_pnl:>14,.2f}")
        lines.append(f"  Realized P&L:       ${account.realized_pnl:>14,.2f}")
    else:
        lines.append("  Could not retrieve account summary")

    positions = executor.get_position_summary()
    open_positions = positions.get('positions', {})
    if open_positions:
        lines.append(f"  Open Positions ({len(open_positions)}):")
        for sym, pos in open_positions.items():
            shares = getattr(pos, 'shares', pos.get('shares', 0) if isinstance(pos, dict) else 0)
            avg_cost = getattr(pos, 'avg_cost', pos.get('avg_cost', 0) if isinstance(pos, dict) else 0)
            mkt_val = getattr(pos, 'market_value', pos.get('market_value', 0) if isinstance(pos, dict) else 0)
            lines.append(f"    {sym:6s}  {shares:>4} shares  avg_cost=${avg_cost:>8.2f}  mkt_val=${mkt_val:>10,.2f}")
    else:
        lines.append("  Open Positions: none")
    lines.append("")
    return "\n".join(lines)


def execute_signal_file(filepath, executor, dry_run=False, default_shares=None,
                        buy_only=False):
    """
    Read and execute all signals in a CSV file.

    Args:
        filepath: Path to signal CSV
        executor: TradeExecutor instance (None if dry_run)
        dry_run: If True, print signals but don't execute
        default_shares: Override shares=0 with this fixed count
        buy_only: If True, skip SELL signals for symbols we don't hold

    Returns:
        list of result dicts for the execution log
    """
    signals = read_signals(filepath)
    if not signals:
        logger.info(f"No signals in {filepath}")
        return []

    # Apply default shares override (e.g. --default-shares 1 for testing)
    if default_shares is not None and default_shares > 0:
        for sig in signals:
            if int(sig.get('shares', 0)) == 0:
                sig['shares'] = default_shares
        logger.info(f"Applied default shares={default_shares} to signals with shares=0")

    # De-duplicate: if multiple strategies signal the same (symbol, action),
    # keep only the first one to prevent double-buying
    seen = set()
    deduped = []
    rejections = []  # Track all rejections for email
    for sig in signals:
        key = (sig['symbol'].upper(), sig['action'].upper())
        if key in seen:
            reason = f"duplicate signal (strategy={sig.get('strategy', 'unknown')})"
            is_new = _is_new_rejection(sig['symbol'], 'duplicate')
            if is_new:
                logger.warning(f"REJECTED {sig['action']} {sig['symbol']}: {reason}")
            else:
                logger.debug(f"De-dup (already notified): {sig['action']} {sig['symbol']}")
            rejections.append({
                **sig,
                'status': 'rejected',
                'rejection_reason': reason,
                'is_new': is_new,
                'executed_at': datetime.now().isoformat(timespec='seconds'),
            })
            continue
        seen.add(key)
        deduped.append(sig)

    if len(deduped) < len(signals):
        logger.info(f"De-duplicated {len(signals)} signals down to {len(deduped)}")
    signals = deduped

    # Buy-only filter: skip SELL signals for symbols we don't actually hold
    if buy_only and not dry_run and executor:
        held_symbols = set()
        try:
            pos_summary = executor.get_position_summary()
            held_symbols = set(pos_summary.get('positions', {}).keys())
        except Exception as e:
            logger.warning(f"Could not fetch positions for buy-only filter: {e}")

        filtered = []
        sell_skipped = 0
        for sig in signals:
            if sig['action'].upper() == 'SELL' and sig['symbol'].upper() not in held_symbols:
                sell_skipped += 1
                continue
            filtered.append(sig)

        if sell_skipped > 0:
            logger.info(f"Buy-only filter: skipped {sell_skipped} SELL signals (no position held), "
                        f"kept {len(filtered)} signals")
        signals = filtered

    logger.info(f"Processing {len(signals)} signal(s) from {filepath}")

    # Pre-execution: verbose account state
    pre_account = None
    if not dry_run and executor:
        pre_account = log_account_state(executor, label="PRE-EXECUTION")

    results = []

    for sig in signals:
        action = sig['action'].upper()
        symbol = sig['symbol'].upper()
        shares = int(sig.get('shares', 0)) or None  # 0 -> None (auto-size)
        signal_price = float(sig.get('price', 0))
        reason = sig.get('reason', '')
        strategy = sig.get('strategy', '')

        label = f"{action} {symbol}"
        if shares:
            label += f" x{shares}"
        if signal_price > 0:
            label += f" @ ${signal_price:.2f}"
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

        logger.info("-" * 50)
        logger.info(f"Executing: {label}")

        try:
            if action == 'BUY':
                result = executor.execute_buy(symbol, shares=shares, reason=reason,
                                              fallback_price=signal_price)
            elif action == 'SELL':
                result = executor.execute_sell(symbol, shares=shares, reason=reason,
                                              fallback_price=signal_price)
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

            if result.success:
                logger.info(f"  FILLED: {symbol} {filled_shares} shares @ ${fill_price:.2f} "
                           f"(total: ${filled_shares * fill_price:,.2f})")
            else:
                # Classify rejection reason for daily de-dup
                reason_type = 'failed'
                if 'spread' in error.lower():
                    reason_type = 'spread'
                elif 'position' in error.lower() or 'already' in error.lower():
                    reason_type = 'position'
                elif 'exposure' in error.lower():
                    reason_type = 'exposure'
                elif 'funds' in error.lower():
                    reason_type = 'funds'
                elif 'daily' in error.lower():
                    reason_type = 'daily_limit'

                is_new = _is_new_rejection(symbol, reason_type)
                if is_new:
                    logger.warning(f"  REJECTED: {symbol} - {error}")
                else:
                    logger.debug(f"  REJECTED (already notified): {symbol} - {error}")

                rejections.append({
                    **sig,
                    'status': 'rejected',
                    'rejection_reason': error,
                    'is_new': is_new,
                    'executed_at': datetime.now().isoformat(timespec='seconds'),
                })

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

    # Post-execution: verbose account state + summary
    if not dry_run and executor:
        post_account = log_account_state(executor, label="POST-EXECUTION")

        # Print execution summary
        filled = [r for r in results if r['status'] == 'filled']
        failed = [r for r in results if r['status'] == 'failed']
        errors = [r for r in results if r['status'] == 'error']
        total_cost = sum(r['fill_price'] * r['filled_shares'] for r in filled)

        logger.info("=" * 70)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Total signals:  {len(results)}")
        logger.info(f"  Filled:         {len(filled)}")
        logger.info(f"  Failed:         {len(failed)}")
        logger.info(f"  Errors:         {len(errors)}")
        logger.info(f"  Total cost:     ${total_cost:,.2f}")
        if filled:
            logger.info(f"  Fills:")
            for r in filled:
                logger.info(f"    {r['action']} {r['symbol']}: {r['filled_shares']} shares @ ${r['fill_price']:.2f} "
                           f"= ${r['filled_shares'] * r['fill_price']:,.2f}")
        if failed:
            logger.info(f"  Failures:")
            for r in failed:
                logger.info(f"    {r['symbol']}: {r['error']}")
        if errors:
            logger.info(f"  Errors:")
            for r in errors:
                logger.info(f"    {r['symbol']}: {r['error']}")
        logger.info("=" * 70)

        # Wait for IBKR to update portfolio state before fetching summary
        import time
        time.sleep(2)

        # Send batch execution summary email (include new rejections only)
        new_rejections = [r for r in rejections if r.get('is_new')]
        _send_execution_summary_email(executor, results, pre_account, post_account,
                                      rejections=new_rejections)

    return results


def _send_execution_summary_email(executor, results, pre_account, post_account,
                                   rejections=None):
    """Send a comprehensive execution summary email."""
    if not executor.notifier:
        return

    rejections = rejections or []
    filled = [r for r in results if r['status'] == 'filled']
    failed = [r for r in results if r['status'] == 'failed']
    errors = [r for r in results if r['status'] == 'error']
    total_cost = sum(r['fill_price'] * r['filled_shares'] for r in filled)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rej_tag = f" | {len(rejections)} rejected" if rejections else ""
    subject = f"[EXECUTION] {len(filled)}/{len(results)} filled | ${total_cost:,.2f} deployed{rej_tag}"

    body_lines = [
        "TRADE EXECUTION REPORT",
        "=" * 50,
        f"Time:           {timestamp}",
        f"Total signals:  {len(results)}",
        f"Filled:         {len(filled)}",
        f"Failed:         {len(failed)}",
        f"Errors:         {len(errors)}",
        f"Total deployed: ${total_cost:,.2f}",
        "",
    ]

    if filled:
        body_lines.append("FILLS")
        body_lines.append("-" * 50)
        for r in filled:
            body_lines.append(
                f"  {r['action']:4s} {r['symbol']:6s}  {r['filled_shares']:>4} shares "
                f"@ ${r['fill_price']:>8.2f}  = ${r['filled_shares'] * r['fill_price']:>10,.2f}"
            )
        body_lines.append("")

    if failed:
        body_lines.append("FAILURES")
        body_lines.append("-" * 50)
        for r in failed:
            body_lines.append(f"  {r['action']:4s} {r['symbol']:6s}  {r.get('error', 'unknown')}")
        body_lines.append("")

    if errors:
        body_lines.append("ERRORS")
        body_lines.append("-" * 50)
        for r in errors:
            body_lines.append(f"  {r['action']:4s} {r['symbol']:6s}  {r.get('error', 'unknown')}")
        body_lines.append("")

    if rejections:
        body_lines.append(f"REJECTIONS ({len(rejections)} new today)")
        body_lines.append("-" * 50)
        for r in rejections:
            body_lines.append(
                f"  {r.get('action', '?'):4s} {r.get('symbol', '?'):6s}  "
                f"{r.get('rejection_reason', 'unknown')}"
            )
        body_lines.append("")

    # Pre-execution account state
    if pre_account:
        body_lines.append("PRE-EXECUTION ACCOUNT")
        body_lines.append("-" * 50)
        body_lines.append(f"  Net Liquidation:  ${pre_account.net_liquidation:>14,.2f}")
        body_lines.append(f"  Total Cash:       ${pre_account.total_cash:>14,.2f}")
        body_lines.append(f"  Available Funds:  ${pre_account.available_funds:>14,.2f}")
        body_lines.append(f"  Buying Power:     ${pre_account.buying_power:>14,.2f}")
        body_lines.append("")

    # Post-execution account state
    if post_account:
        body_lines.append("POST-EXECUTION ACCOUNT")
        body_lines.append("-" * 50)
        body_lines.append(f"  Net Liquidation:  ${post_account.net_liquidation:>14,.2f}")
        body_lines.append(f"  Total Cash:       ${post_account.total_cash:>14,.2f}")
        body_lines.append(f"  Available Funds:  ${post_account.available_funds:>14,.2f}")
        body_lines.append(f"  Buying Power:     ${post_account.buying_power:>14,.2f}")
        body_lines.append(f"  Gross Positions:  ${post_account.gross_position_value:>14,.2f}")
        body_lines.append(f"  Unrealized P&L:   ${post_account.unrealized_pnl:>14,.2f}")
        body_lines.append("")

    # Post-execution portfolio
    body_lines.append(format_account_state_text(executor, label="CURRENT PORTFOLIO"))

    body = "\n".join(body_lines)
    executor.notifier._send_email(subject, body)


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
                    dry_run=False, pattern="pending_orders*.csv", default_shares=None,
                    buy_only=False):
    """
    Poll a directory for new signal files and execute them.

    Args:
        signals_dir: Directory to watch
        executor: TradeExecutor instance
        archive_dir: Where to move processed files
        interval_seconds: Seconds between polls
        dry_run: If True, print but don't execute
        pattern: Glob pattern for signal files
        buy_only: If True, skip SELL signals for symbols we don't hold
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
                results = execute_signal_file(filepath, executor, dry_run=dry_run,
                                              default_shares=default_shares,
                                              buy_only=buy_only)
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
    parser.add_argument("--default-shares", type=int, default=None,
                        help="Override shares=0 signals with this fixed share count (e.g. 1 for testing)")
    parser.add_argument("--market-orders", action="store_true",
                        help="Use market orders instead of limit orders")
    parser.add_argument("--buy-only", action="store_true",
                        help="Skip SELL signals for symbols we don't hold (still sell owned positions)")
    parser.add_argument("--require-market-hours", action="store_true",
                        help="Only execute during US market hours (9:30-16:00 ET, Mon-Fri)")

    args = parser.parse_args()

    # Validate args
    if not args.signals and not args.watch:
        parser.error("Must specify --signals (one-shot) or --watch (poll mode)")

    archive_dir = args.archive_dir or os.path.join(args.signals_dir, "executed")

    # Market hours check
    if args.require_market_hours and not is_market_open():
        now_et = datetime.now(ZoneInfo("America/New_York"))
        logger.warning(f"Market is CLOSED (ET: {now_et.strftime('%A %H:%M')}). "
                       f"Use without --require-market-hours to execute anyway.")
        sys.exit(0)

    # Dry run: no broker connection needed (buy_only filter needs executor, so skipped in dry-run)
    if args.dry_run:
        if args.signals:
            results = execute_signal_file(args.signals, executor=None, dry_run=True,
                                          default_shares=args.default_shares,
                                          buy_only=args.buy_only)
            if results:
                print(f"\n{len(results)} signal(s) would be executed.")
        elif args.watch:
            watch_directory(args.signals_dir, executor=None, archive_dir=archive_dir,
                            interval_seconds=args.interval, dry_run=True,
                            default_shares=args.default_shares,
                            buy_only=args.buy_only)
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
        use_market_orders=args.market_orders,
    )

    # Execute
    with TradeExecutor(ibkr_config, trading_config) as executor:
        if args.signals:
            # One-shot mode
            results = execute_signal_file(args.signals, executor,
                                          default_shares=args.default_shares,
                                          buy_only=args.buy_only)
            archive_signal_file(args.signals, results, archive_dir)
        elif args.watch:
            # Watch mode
            watch_directory(args.signals_dir, executor, archive_dir,
                            interval_seconds=args.interval,
                            default_shares=args.default_shares,
                            buy_only=args.buy_only)


if __name__ == "__main__":
    main()
