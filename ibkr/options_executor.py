"""
Options execution module for IBKR trading.

Translates stock BUY/SELL signals into options orders:
  - BUY signal  -> buy a call, 1 strike OTM, next monthly expiration
  - SELL signal -> buy a put,  1 strike OTM, next monthly expiration

Always buys to open — never sells naked options.

Usage:
    from ibkr.options_executor import execute_option_signal
    result = execute_option_signal(executor, signal_dict)
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

from ib_insync import Option as IBOption, MarketOrder, LimitOrder

# Ensure project root on path for rt_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_data.rt_utils import get_options_snapshot

logger = logging.getLogger(__name__)

# Guardrail defaults
MAX_SPREAD_PCT = 0.05       # 5% max bid-ask spread (percentage)
MAX_SPREAD_DOLLAR = 0.20    # $0.20 per share -> $20 per 100-lot contract
MIN_BID = 0.05              # Minimum bid price (filter penny options)
MIN_OPEN_INTEREST = 10      # Minimum open interest
MIN_VOLUME = 1              # Minimum daily volume
MAX_CONTRACTS = 20           # Hard cap on auto-sized contracts per signal
DEFAULT_MONTHS_OUT = 1      # Next monthly expiration
NUM_STRIKES = 3             # Strikes above/below ATM to fetch
REPRICE_SPREAD_PCT = 0.15   # After timeout, bump limit by 15% of spread toward ask


def select_option_contract(symbol: str, action: str, stock_price: float,
                           months_out: int = DEFAULT_MONTHS_OUT):
    """
    Fetch the options chain and select the target contract.

    Args:
        symbol: Stock ticker
        action: 'BUY' or 'SELL' (from the stock signal)
        stock_price: Current stock price (for fallback/logging)
        months_out: Months out for expiration

    Returns:
        Tuple of (option_row as dict, quote_dict, full options DataFrame)
        or (None, None, None) on failure with logged reason.
    """
    # BUY signal -> call; SELL signal -> put
    option_type = 'call' if action == 'BUY' else 'put'

    try:
        quote, options_df = get_options_snapshot(
            symbol, months_out=months_out,
            num_strikes=NUM_STRIKES, option_type=option_type
        )
    except Exception as e:
        logger.error(f"Failed to fetch options chain for {symbol}: {e}")
        return None, None, None

    if options_df is None or options_df.empty:
        logger.warning(f"No {option_type} options found for {symbol}")
        return None, None, None

    current_price = quote.get('price', stock_price)

    # Pick 1 strike OTM
    if option_type == 'call':
        # First strike strictly above current price
        otm = options_df[options_df['strike'] > current_price].sort_values('strike')
    else:
        # First strike strictly below current price
        otm = options_df[options_df['strike'] < current_price].sort_values('strike', ascending=False)

    if otm.empty:
        logger.warning(f"No OTM {option_type} strikes found for {symbol} (price=${current_price:.2f})")
        return None, quote, options_df

    target = otm.iloc[0]
    return target.to_dict(), quote, options_df


def validate_option(option: dict, symbol: str) -> Optional[str]:
    """
    Validate an option contract against guardrails.

    Returns:
        None if valid, or a rejection reason string.
    """
    bid = option.get('bid', 0) or 0
    ask = option.get('ask', 0) or 0
    mid = option.get('mid', 0) or 0
    oi = option.get('open_interest', 0) or 0
    vol = option.get('volume', 0) or 0

    # Must have valid bid/ask
    if ask <= 0:
        return f"No valid ask price (ask={ask})"

    # Minimum bid filter — reject penny options
    if bid < MIN_BID:
        return (f"Bid too low: ${bid:.2f} < ${MIN_BID:.2f} "
                f"(ask={ask:.2f}) — filtered as penny option")

    if mid <= 0:
        mid = (bid + ask) / 2
        if mid <= 0:
            return f"No valid mid price (bid={bid}, ask={ask})"

    # Spread check — reject only when BOTH percentage AND dollar spread are wide.
    # This allows cheap options with high % spread but tiny absolute spread
    # (e.g. bid=0.15/ask=0.17 → 12% but only $2 per contract).
    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
    spread_dollar = ask - bid
    if spread_pct >= MAX_SPREAD_PCT and spread_dollar >= MAX_SPREAD_DOLLAR:
        return (f"Spread too wide: {spread_pct:.1%} (>={MAX_SPREAD_PCT:.0%}) "
                f"and ${spread_dollar:.2f}/share (>=${MAX_SPREAD_DOLLAR:.2f}, "
                f"=${spread_dollar * 100:.0f}/contract) "
                f"(bid={bid:.2f}, ask={ask:.2f}, mid={mid:.2f})")

    # Liquidity check
    if oi < MIN_OPEN_INTEREST:
        return f"Open interest too low: {oi} < {MIN_OPEN_INTEREST}"

    if vol < MIN_VOLUME:
        return f"Volume too low: {vol} < {MIN_VOLUME}"

    return None


def calculate_option_contracts(executor, mid_price: float) -> int:
    """
    Auto-size option contracts using the same position-sizing rules as stocks.

    Uses: target_value = net_liquidation * position_size (default 2%)
    Cost per contract = mid_price * 100 (each contract = 100 shares)

    Returns number of contracts (minimum 1 if affordable).
    """
    if executor is None or mid_price <= 0:
        return 1

    try:
        account = executor.get_account_summary()
        if account is None or account.net_liquidation <= 0:
            return 1

        config = executor.trading_config
        contract_cost = mid_price * 100

        # Same formula as risk_manager.calculate_position_size
        target_value = account.net_liquidation * config.position_size
        target_value = min(target_value, config.max_position_value)
        target_value = min(target_value, account.available_funds)

        contracts = int(target_value / contract_cost)

        # Enforce minimum order value
        if contracts > 0 and contracts * contract_cost < config.min_order_value:
            return 0

        # Hard cap to prevent absurd quantities on cheap options
        contracts = min(contracts, MAX_CONTRACTS)

        return max(1, contracts)
    except Exception as e:
        logger.warning(f"Options auto-size failed ({e}), defaulting to 1 contract")
        return 1


def execute_option_signal(executor, sig: dict, dry_run: bool = False,
                          use_market_orders: bool = False) -> dict:
    """
    Execute a single stock signal as an options trade.

    Args:
        executor: TradeExecutor instance (has .order_manager, .connection)
        sig: Signal dict with keys: action, symbol, shares, price, strategy, reason
        dry_run: If True, log what would happen but don't submit
        use_market_orders: If True, use market orders; else mid-point limit

    Returns:
        Result dict compatible with execute_signals.py email/archive format.
    """
    action = sig['action'].upper()
    symbol = sig['symbol'].upper()
    raw_shares = int(sig.get('shares', 0))
    signal_price = float(sig.get('price', 0))
    option_type = 'call' if action == 'BUY' else 'put'
    timestamp = datetime.now().isoformat(timespec='seconds')

    base_result = {
        **sig,
        'contract_type': option_type,
        'strike': 0.0,
        'expiration': '',
        'premium': 0.0,
        'executed_at': timestamp,
    }

    # 1. Select the option contract
    logger.info(f"Selecting {option_type} for {symbol} ({action} signal, price=${signal_price:.2f})")
    option, quote, options_df = select_option_contract(symbol, action, signal_price)

    if option is None:
        reason = f"No suitable {option_type} option found for {symbol}"
        logger.warning(f"  REJECTED: {reason}")
        return {**base_result, 'status': 'failed', 'fill_price': 0.0,
                'filled_shares': 0, 'error': reason}

    strike = option['strike']
    expiration = option['expiration']
    bid = option.get('bid', 0) or 0
    ask = option.get('ask', 0) or 0
    mid = option.get('mid', 0) or ((bid + ask) / 2)
    oi = option.get('open_interest', 0) or 0
    vol = option.get('volume', 0) or 0
    iv = option.get('implied_volatility', 0) or 0
    delta = option.get('delta', 0) or 0

    base_result['strike'] = strike
    base_result['expiration'] = expiration
    base_result['premium'] = mid

    # Auto-size contracts if shares=0 (same position-sizing as stocks)
    if raw_shares == 0:
        contracts = calculate_option_contracts(executor, mid)
        logger.info(f"  Auto-sized: {contracts} contract(s) "
                     f"(${mid:.2f} x 100 = ${mid * 100:.2f}/contract)")
    else:
        contracts = raw_shares

    logger.info(f"  Selected: {symbol} {expiration} {strike} {option_type.upper()}")
    logger.info(f"  Bid={bid:.2f} Ask={ask:.2f} Mid={mid:.2f} OI={oi} Vol={vol} IV={iv:.2f} Delta={delta:.3f}")
    logger.info(f"  Contracts={contracts} (total ~${contracts * mid * 100:,.2f})")

    # 2. Validate guardrails
    rejection = validate_option(option, symbol)
    if rejection:
        logger.warning(f"  REJECTED: {rejection}")
        return {**base_result, 'status': 'failed', 'fill_price': 0.0,
                'filled_shares': 0, 'error': rejection}

    # 3. Dry run: log and return
    if dry_run:
        logger.info(f"  [DRY RUN] Would BUY {contracts} x {symbol} {expiration} "
                     f"{strike} {option_type.upper()} @ ~${mid:.2f} "
                     f"(total ~${contracts * mid * 100:.2f})")
        return {**base_result, 'status': 'dry_run', 'fill_price': 0.0,
                'filled_shares': 0, 'error': ''}

    # 4. Build ib_insync Option contract
    # Expiration format for IBKR: YYYYMMDD
    ibkr_exp = expiration.replace('-', '')
    right = 'C' if option_type == 'call' else 'P'
    contract = IBOption(symbol, ibkr_exp, strike, right, 'SMART', currency='USD')

    try:
        executor.order_manager.ib.qualifyContracts(contract)
    except Exception as e:
        reason = f"Failed to qualify option contract: {e}"
        logger.error(f"  {reason}")
        return {**base_result, 'status': 'error', 'fill_price': 0.0,
                'filled_shares': 0, 'error': reason}

    # 5. Create order — always BUY to open
    if use_market_orders:
        order = MarketOrder('BUY', contracts)
    else:
        # Mid-point limit order: better fill than ask, reasonable chance of fill
        limit_price = round(mid, 2)
        order = LimitOrder('BUY', contracts, limit_price)
        logger.info(f"  Mid-point limit order @ ${limit_price:.2f} (bid={bid:.2f}, ask={ask:.2f})")

    order.tif = 'DAY'
    acct = executor.order_manager.target_account
    if acct:
        order.account = acct

    # 6. Submit
    try:
        trade = executor.order_manager.ib.placeOrder(contract, order)
        order_id = trade.order.orderId
        logger.info(f"  Order submitted: BUY {contracts} x {symbol} {expiration} "
                     f"{strike}{right} (order_id={order_id})")

        # Wait for fill with repricing: first wait at mid, then bump toward ask
        executor.order_manager.ib.sleep(2)  # Let IB process
        timeout = executor.order_manager.config.order_timeout
        import time
        start = time.time()
        repriced = False

        while time.time() - start < timeout:
            executor.order_manager.ib.sleep(0.5)
            if trade.isDone():
                break

        # If not filled after first timeout, reprice closer to ask and wait again
        if not trade.isDone() and not use_market_orders and ask > bid:
            spread = ask - bid
            bump = round(spread * REPRICE_SPREAD_PCT, 2)
            new_limit = round(mid + bump, 2)
            logger.info(f"  Repricing: mid=${mid:.2f} + 15% of spread (${spread:.2f}) "
                        f"= ${new_limit:.2f} (was ${limit_price:.2f})")
            try:
                trade.order.lmtPrice = new_limit
                executor.order_manager.ib.placeOrder(contract, trade.order)
                repriced = True

                # Wait another timeout period
                start2 = time.time()
                while time.time() - start2 < timeout:
                    executor.order_manager.ib.sleep(0.5)
                    if trade.isDone():
                        break
            except Exception as e:
                logger.warning(f"  Reprice failed: {e}")

        fill_price = trade.orderStatus.avgFillPrice or 0.0
        filled = int(trade.orderStatus.filled or 0)
        status_str = trade.orderStatus.status

        if status_str == 'Filled':
            total_premium = filled * fill_price * 100
            reprice_tag = " (after reprice)" if repriced else ""
            logger.info(f"  FILLED: {filled} contracts @ ${fill_price:.2f} "
                        f"(premium=${total_premium:,.2f}){reprice_tag}")
            return {**base_result, 'status': 'filled', 'fill_price': fill_price,
                    'filled_shares': filled, 'error': ''}
        else:
            # Cancel unfilled order
            if not trade.isDone():
                try:
                    executor.order_manager.ib.cancelOrder(trade.order)
                    executor.order_manager.ib.sleep(1)
                except Exception:
                    pass
            reason = f"Order not filled (status={status_str})"
            logger.warning(f"  {reason}")
            return {**base_result, 'status': 'failed', 'fill_price': fill_price,
                    'filled_shares': filled, 'error': reason}

    except Exception as e:
        reason = f"Order submission failed: {e}"
        logger.error(f"  {reason}")
        return {**base_result, 'status': 'error', 'fill_price': 0.0,
                'filled_shares': 0, 'error': reason}
