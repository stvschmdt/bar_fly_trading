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
from ibkr.options_executor import execute_option_signal
from ibkr.position_ledger import PositionLedger

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


def _parse_exit_param(value, default=None, as_int=False):
    """Parse an exit param from signal CSV (may be empty, NaN, or numeric)."""
    if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return int(float(value)) if as_int else float(value)
    except (ValueError, TypeError):
        return default


def _submit_exit_brackets(executor, symbol, shares, fill_price, sl_pct, tp_pct):
    """Submit stop-loss and take-profit bracket orders after a BUY fill."""
    try:
        result = executor.order_manager.submit_bracket_order(
            symbol=symbol,
            shares=shares,
            entry_price=fill_price,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )
        if result:
            stop_result, profit_result = result
            return stop_result, profit_result
    except Exception as e:
        logger.error(f"Failed to submit bracket orders for {symbol}: {e}")
    return None


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
                        buy_only=False, options_mode=False):
    """
    Read and execute all signals in a CSV file.

    Args:
        filepath: Path to signal CSV
        executor: TradeExecutor instance (None if dry_run)
        dry_run: If True, print signals but don't execute
        default_shares: Override shares=0 with this fixed count
        buy_only: If True, skip SELL signals for symbols we don't hold
        options_mode: If True, default instrument_type for signals missing the field

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
        unit = "contracts" if options_mode else "shares"
        logger.info(f"Applied default {unit}={default_shares} to signals with {unit}=0")

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

        # Determine instrument type: per-signal field overrides global flag
        instrument_type = sig.get('instrument_type', 'stock')
        if instrument_type in ('', 'stock') and options_mode:
            instrument_type = 'option'
        is_option = (instrument_type == 'option')

        if dry_run and not is_option:
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

        # Options: route through options_executor
        if is_option:
            # Check position ledger for existing option position before executing
            if action == 'BUY' and not dry_run:
                try:
                    ledger = PositionLedger()
                    ledger.load()
                    existing = ledger.get_all_positions()
                    # Check if any open position matches this symbol
                    already_held = any(
                        pos.get('symbol', '').upper() == symbol
                        and pos.get('instrument_type') == 'option'
                        for pos in existing.values()
                    )
                    if already_held:
                        reason_msg = f"Already have option position in {symbol}"
                        is_new = _is_new_rejection(symbol, 'position')
                        if is_new:
                            logger.warning(f"REJECTED {action} {symbol}: {reason_msg}")
                        rejections.append({
                            **sig,
                            'status': 'rejected',
                            'rejection_reason': reason_msg,
                            'is_new': is_new,
                            'executed_at': datetime.now().isoformat(timespec='seconds'),
                        })
                        results.append({
                            **sig,
                            'status': 'failed',
                            'fill_price': 0.0,
                            'filled_shares': 0,
                            'error': reason_msg,
                            'executed_at': datetime.now().isoformat(timespec='seconds'),
                        })
                        continue
                except Exception as e:
                    logger.warning(f"Could not check ledger for {symbol}: {e}")

            logger.info("-" * 50)
            opt_label = f"[OPTIONS{'|DRY' if dry_run else ''}] {label}"
            logger.info(f"Executing: {opt_label}")
            use_mkt = executor.trading_config.use_market_orders if executor else False
            opt_result = execute_option_signal(
                executor, sig, dry_run=dry_run, use_market_orders=use_mkt
            )
            results.append(opt_result)

            # Per-fill email notification for options
            if opt_result.get('status') == 'filled' and executor and executor.notifier:
                opt_fill = opt_result.get('fill_price', 0)
                opt_qty = opt_result.get('filled_shares', 0)
                opt_total = opt_fill * opt_qty * 100
                opt_strike = opt_result.get('strike', '')
                opt_exp = opt_result.get('expiration', '')
                opt_ctype = opt_result.get('contract_type', '').upper()
                executor.notifier._send_email(
                    f"[FILL] BUY {opt_qty} {symbol} {opt_exp} {opt_strike} {opt_ctype} @ ${opt_fill:.2f}",
                    f"OPTIONS FILL\n{'=' * 40}\n"
                    f"Symbol:     {symbol}\n"
                    f"Contract:   {opt_exp} {opt_strike} {opt_ctype}\n"
                    f"Contracts:  {opt_qty}\n"
                    f"Premium:    ${opt_fill:.2f} per share\n"
                    f"Total cost: ${opt_total:,.2f}\n"
                    f"Strategy:   {sig.get('strategy', '')}\n"
                    f"Reason:     {sig.get('reason', '')}\n"
                    f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

            # Record options fill to position ledger (same as stocks)
            if opt_result.get('status') == 'filled' and action == 'BUY':
                opt_fill = opt_result.get('fill_price', 0)
                opt_qty = opt_result.get('filled_shares', 0)
                opt_strike_val = opt_result.get('strike', 0)
                opt_exp_val = opt_result.get('expiration', '')
                opt_ctype_val = opt_result.get('contract_type', '')

                sl_pct = _parse_exit_param(sig.get('stop_loss_pct'), default=-0.08)
                tp_pct = _parse_exit_param(sig.get('take_profit_pct'), default=0.15)
                ts_pct = _parse_exit_param(sig.get('trailing_stop_pct'), default=None)
                ta_pct = _parse_exit_param(sig.get('trailing_activation_pct'), default=0.0)
                max_hold = _parse_exit_param(sig.get('max_hold_days'), default=20, as_int=True)

                try:
                    ledger = PositionLedger()
                    ledger.load()
                    ledger_key = ledger.add_position(
                        symbol=symbol,
                        entry_price=opt_fill,
                        entry_date=datetime.now().strftime('%Y-%m-%d'),
                        shares=opt_qty,
                        strategy=sig.get('strategy', ''),
                        stop_loss_pct=sl_pct,
                        take_profit_pct=tp_pct,
                        trailing_stop_pct=ts_pct,
                        max_hold_days=max_hold,
                        stop_order_id=-1,
                        profit_order_id=-1,
                        parent_order_id=-1,
                        trailing_activation_pct=ta_pct,
                        instrument_type='option',
                        contract_type=opt_ctype_val,
                        strike=float(opt_strike_val) if opt_strike_val else 0.0,
                        expiration=str(opt_exp_val),
                    )
                    ledger.save()
                    logger.info(f"  LEDGER: Options position recorded as {ledger_key}")
                except Exception as e:
                    logger.error(f"  Failed to save options position to ledger: {e}")

            # Track rejections for email
            if opt_result.get('status') == 'failed':
                reason_type = 'options'
                is_new = _is_new_rejection(symbol, reason_type)
                opt_result['is_new'] = is_new
                rejections.append({
                    **opt_result,
                    'rejection_reason': opt_result.get('error', 'unknown'),
                    'is_new': is_new,
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

                # Place bracket orders (stop-loss + take-profit) after BUY fill
                if action == 'BUY' and fill_price > 0 and filled_shares > 0:
                    sl_pct = _parse_exit_param(sig.get('stop_loss_pct'), default=-0.08)
                    tp_pct = _parse_exit_param(sig.get('take_profit_pct'), default=0.15)
                    ts_pct = _parse_exit_param(sig.get('trailing_stop_pct'), default=None)
                    ta_pct = _parse_exit_param(sig.get('trailing_activation_pct'), default=0.0)
                    max_hold = _parse_exit_param(sig.get('max_hold_days'), default=20, as_int=True)

                    bracket = _submit_exit_brackets(
                        executor, symbol, filled_shares, fill_price, sl_pct, tp_pct)

                    stop_oid = -1
                    profit_oid = -1
                    if bracket:
                        stop_res, profit_res = bracket
                        stop_oid = stop_res.order_id
                        profit_oid = profit_res.order_id
                        logger.info(
                            f"  BRACKETS: {symbol} stop={stop_oid} "
                            f"@ ${round(fill_price * (1 + sl_pct), 2)}, "
                            f"profit={profit_oid} "
                            f"@ ${round(fill_price * (1 + tp_pct), 2)}")
                    else:
                        logger.warning(f"  WARNING: No bracket orders for {symbol} "
                                      f"— exit monitor will handle SL/TP")

                    # Always record to position ledger (exit monitor handles
                    # SL/TP via software polling if brackets failed)
                    try:
                        ledger = PositionLedger()
                        ledger.load()
                        ledger.add_position(
                            symbol=symbol,
                            entry_price=fill_price,
                            entry_date=datetime.now().strftime('%Y-%m-%d'),
                            shares=filled_shares,
                            strategy=sig.get('strategy', ''),
                            stop_loss_pct=sl_pct,
                            take_profit_pct=tp_pct,
                            trailing_stop_pct=ts_pct,
                            max_hold_days=max_hold,
                            stop_order_id=stop_oid,
                            profit_order_id=profit_oid,
                            parent_order_id=result.order.order_id if result.order else -1,
                            trailing_activation_pct=ta_pct,
                        )
                        ledger.save()
                        logger.info(f"  LEDGER: Stock position recorded for {symbol}")
                    except Exception as e:
                        logger.error(f"  Failed to save position ledger for {symbol}: {e}")
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
                is_option = bool(r.get('contract_type'))
                unit = "contracts" if is_option else "shares"
                multiplier = 100 if is_option else 1
                fill = r['fill_price']
                qty = r['filled_shares']
                total = fill * qty * multiplier
                sl_pct = _parse_exit_param(r.get('stop_loss_pct'), default=-0.08)
                tp_pct = _parse_exit_param(r.get('take_profit_pct'), default=0.15)
                tp_dollar = fill * tp_pct * qty * multiplier
                sl_dollar = fill * sl_pct * qty * multiplier
                logger.info(f"    {r['action']} {r['symbol']}: {qty} {unit} @ ${fill:.2f} "
                           f"= ${total:,.2f}  PLR (+${tp_dollar:,.0f}, -${abs(sl_dollar):,.0f})")
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
            is_option = bool(r.get('contract_type'))
            unit = "contracts" if is_option else "shares"
            multiplier = 100 if is_option else 1
            detail = ""
            if is_option:
                detail = f"  {r.get('expiration', '')} {r.get('strike', '')} {r['contract_type'].upper()}"
            fill = r['fill_price']
            qty = r['filled_shares']
            total = fill * qty * multiplier

            # PLR: profit/loss range from exit params
            sl_pct = _parse_exit_param(r.get('stop_loss_pct'), default=-0.08)
            tp_pct = _parse_exit_param(r.get('take_profit_pct'), default=0.15)
            tp_dollar = fill * tp_pct * qty * multiplier
            sl_dollar = fill * sl_pct * qty * multiplier
            plr = f"PLR (+${tp_dollar:,.0f}, -${abs(sl_dollar):,.0f})"

            body_lines.append(
                f"  {r['action']:4s} {r['symbol']:6s}  {qty:>4} {unit}"
                f"{detail}"
                f"  @ ${fill:>8.2f}  = ${total:>10,.2f}  {plr}"
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

    # Archive the original signal file (may already be removed by scan loop)
    archive_path = os.path.join(archive_dir, f"{base}_{ts}.csv")
    if os.path.exists(filepath):
        shutil.move(filepath, archive_path)
        logger.info(f"Archived signal file to {archive_path}")
    else:
        logger.debug(f"Signal file already removed (archived by scan loop): {filepath}")

    # Write execution results alongside
    if results:
        results_path = os.path.join(archive_dir, f"{base}_{ts}_results.csv")
        pd.DataFrame(results).to_csv(results_path, index=False)
        logger.info(f"Execution results written to {results_path}")


def watch_directory(signals_dir, executor, archive_dir, interval_seconds=30,
                    dry_run=False, pattern="pending_orders*.csv", default_shares=None,
                    buy_only=False, options_mode=False):
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
        options_mode: If True, execute as options
    """
    import glob

    logger.info(f"Watching {signals_dir} for signal files (every {interval_seconds}s)...")
    logger.info(f"Pattern: {pattern}")
    if options_mode:
        logger.info("OPTIONS MODE active")
    logger.info("Press Ctrl+C to stop.\n")

    try:
        while True:
            matches = sorted(glob.glob(os.path.join(signals_dir, pattern)))

            for filepath in matches:
                logger.info(f"\nFound signal file: {filepath}")
                results = execute_signal_file(filepath, executor, dry_run=dry_run,
                                              default_shares=default_shares,
                                              buy_only=buy_only,
                                              options_mode=options_mode)
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
    parser.add_argument("--position-size", type=float, default=0.02,
                        help="Position size as fraction of portfolio (default: 0.02)")
    parser.add_argument("--max-positions", type=int, default=20,
                        help="Maximum concurrent positions (default: 20)")
    parser.add_argument("--max-daily-loss", type=float, default=5000,
                        help="Maximum daily loss in dollars (default: 5000)")
    parser.add_argument("--default-shares", type=int, default=None,
                        help="Override shares=0 signals with this fixed share count (e.g. 1 for testing)")
    parser.add_argument("--market-orders", action="store_true",
                        help="Use market orders for all order types")
    parser.add_argument("--stock-limit-orders", action="store_true",
                        help="Use limit orders for stocks (overrides default market)")
    parser.add_argument("--buy-only", action="store_true",
                        help="Skip SELL signals for symbols we don't hold (still sell owned positions)")
    parser.add_argument("--options", action="store_true",
                        help="Execute as options: BUY signal -> buy call OTM, SELL signal -> buy put OTM")
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
    # Options dry-run also works without IBKR (fetches chain from AlphaVantage only)
    if args.dry_run:
        if args.signals:
            results = execute_signal_file(args.signals, executor=None, dry_run=True,
                                          default_shares=args.default_shares,
                                          buy_only=args.buy_only,
                                          options_mode=args.options)
            if results:
                print(f"\n{len(results)} signal(s) would be executed.")
        elif args.watch:
            watch_directory(args.signals_dir, executor=None, archive_dir=archive_dir,
                            interval_seconds=args.interval, dry_run=True,
                            default_shares=args.default_shares,
                            buy_only=args.buy_only,
                            options_mode=args.options)
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
        stock_market_orders=not args.stock_limit_orders,  # Stocks default to market
    )

    if args.options:
        logger.info("OPTIONS MODE: signals will be executed as option contracts")

    # Execute
    with TradeExecutor(ibkr_config, trading_config) as executor:
        if args.signals:
            # One-shot mode
            results = execute_signal_file(args.signals, executor,
                                          default_shares=args.default_shares,
                                          buy_only=args.buy_only,
                                          options_mode=args.options)
            archive_signal_file(args.signals, results, archive_dir)
        elif args.watch:
            # Watch mode
            watch_directory(args.signals_dir, executor, archive_dir,
                            interval_seconds=args.interval,
                            default_shares=args.default_shares,
                            buy_only=args.buy_only,
                            options_mode=args.options)


if __name__ == "__main__":
    main()
