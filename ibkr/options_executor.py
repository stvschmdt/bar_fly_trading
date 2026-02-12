"""
Options execution module for IBKR trading.

Translates stock BUY/SELL signals into options orders:
  - BUY signal  -> buy a call, 1 strike OTM, next monthly expiration
  - SELL signal -> buy a put,  1 strike OTM, next monthly expiration

Always buys to open — never sells naked options.

Flow:
  1. AV HISTORICAL_OPTIONS fetch → select strike (closest to money, tightest spread)
  2. Validate guardrails (spread, OI, volume)
  3. Qualify contract with IBKR, get live/delayed bid-ask from IBKR
  4. Place limit order at IBKR mid, reprice toward ask if not filled

Uses AV for strike selection, IBKR delayed data for live pricing.

Usage:
    from ibkr.options_executor import execute_option_signal
    result = execute_option_signal(executor, signal_dict)
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

from ib_insync import Option as IBOption, MarketOrder, LimitOrder, util

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

    # Find closest OTM and ITM strikes, pick whichever has tighter spread
    if option_type == 'call':
        otm = options_df[options_df['strike'] > current_price].sort_values('strike')
        itm = options_df[options_df['strike'] <= current_price].sort_values('strike', ascending=False)
    else:
        otm = options_df[options_df['strike'] < current_price].sort_values('strike', ascending=False)
        itm = options_df[options_df['strike'] >= current_price].sort_values('strike')

    otm_row = otm.iloc[0] if not otm.empty else None
    itm_row = itm.iloc[0] if not itm.empty else None

    if otm_row is None and itm_row is None:
        logger.warning(f"No {option_type} strikes found for {symbol} (price=${current_price:.2f})")
        return None, quote, options_df

    def _spread(row):
        if row is None:
            return float('inf')
        b = row.get('bid', 0) or 0
        a = row.get('ask', 0) or 0
        return (a - b) if a > b else float('inf')

    otm_spread = _spread(otm_row)
    itm_spread = _spread(itm_row)

    if otm_row is None:
        # No OTM strike available — reject rather than silently opening ITM
        logger.warning(f"No OTM {option_type} strike for {symbol} (price=${current_price:.2f})")
        return None, quote, options_df

    if itm_row is not None:
        if itm_spread < otm_spread:
            target = itm_row
            tag = "ITM (tighter spread)"
        else:
            target = otm_row
            tag = "OTM"
        logger.info(f"  Strike selection: {symbol} price=${current_price:.2f} → "
                    f"OTM ${otm_row['strike']} spread=${otm_spread:.2f}, "
                    f"ITM ${itm_row['strike']} spread=${itm_spread:.2f} → chose {tag}")
    else:
        target = otm_row
        logger.info(f"  Strike selection: {symbol} → OTM (only) ${target['strike']}")

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
    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
    spread_dollar = ask - bid
    if spread_pct >= MAX_SPREAD_PCT and spread_dollar >= MAX_SPREAD_DOLLAR:
        return (f"Spread too wide: {spread_pct:.1%} (>={MAX_SPREAD_PCT:.0%}) "
                f"and ${spread_dollar:.2f}/share (>={MAX_SPREAD_DOLLAR:.2f}, "
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
    """
    if executor is None or mid_price <= 0:
        return 1

    try:
        account = executor.get_account_summary()
        if account is None or account.net_liquidation <= 0:
            return 1

        config = executor.trading_config
        contract_cost = mid_price * 100

        target_value = account.net_liquidation * config.position_size
        target_value = min(target_value, config.max_position_value)
        target_value = min(target_value, account.available_funds)

        contracts = int(target_value / contract_cost)

        if contracts > 0 and contracts * contract_cost < config.min_order_value:
            return 0

        contracts = min(contracts, MAX_CONTRACTS)
        return max(1, contracts)
    except Exception as e:
        logger.warning(f"Options auto-size failed ({e}), defaulting to 1 contract")
        return 1


def get_ibkr_option_quote(ib, contract, timeout: float = 5.0) -> Optional[dict]:
    """
    Get live or delayed bid/ask from IBKR for a qualified option contract.

    Tries live snapshot first; falls back to delayed data if no subscription.

    Returns:
        Dict with bid, ask, mid, last, delayed flag — or None.
    """
    try:
        # Try live snapshot
        ticker = ib.reqMktData(contract, snapshot=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            ib.sleep(0.3)
            bid = ticker.bid if not util.isNan(ticker.bid) else None
            ask = ticker.ask if not util.isNan(ticker.ask) else None
            if bid is not None and ask is not None:
                break

        ib.cancelMktData(contract)

        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            last = ticker.last if not util.isNan(ticker.last) else mid
            return {'bid': bid, 'ask': ask, 'mid': mid, 'last': last, 'delayed': False}

        # Fallback: delayed data (no paid subscription needed)
        logger.info("  Live options data unavailable, requesting delayed...")
        ib.reqMarketDataType(3)  # 3 = delayed
        ticker = ib.reqMktData(contract)
        ib.sleep(0.5)

        deadline = time.time() + timeout
        bid = ask = last = None
        while time.time() < deadline:
            ib.sleep(0.3)
            bid = ticker.bid if not util.isNan(ticker.bid) else None
            ask = ticker.ask if not util.isNan(ticker.ask) else None
            last = ticker.last if not util.isNan(ticker.last) else None
            if bid is not None and ask is not None:
                break
            # Check delayed-specific fields
            if hasattr(ticker, 'delayedBid') and not util.isNan(ticker.delayedBid):
                bid = ticker.delayedBid
            if hasattr(ticker, 'delayedAsk') and not util.isNan(ticker.delayedAsk):
                ask = ticker.delayedAsk
            if hasattr(ticker, 'delayedLast') and not util.isNan(ticker.delayedLast):
                last = ticker.delayedLast
            if bid is not None and ask is not None:
                break

        ib.cancelMktData(contract)
        ib.reqMarketDataType(1)  # Reset to live

        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            logger.info(f"  Delayed quote: bid=${bid:.2f} ask=${ask:.2f} mid=${mid:.2f}")
            return {'bid': bid, 'ask': ask, 'mid': mid, 'last': last or mid, 'delayed': True}

        # Last resort: use last trade price
        if last is not None and last > 0:
            logger.info(f"  Using last trade price: ${last:.2f}")
            return {'bid': last, 'ask': last, 'mid': last, 'last': last, 'delayed': True}

        return None

    except Exception as e:
        logger.warning(f"Failed to get IBKR option quote: {e}")
        try:
            ib.reqMarketDataType(1)
        except Exception:
            pass
        return None


def execute_option_signal(executor, sig: dict, dry_run: bool = False,
                          use_market_orders: bool = False) -> dict:
    """
    Execute a single stock signal as an options trade.

    Flow:
      1. AV historical options → select strike
      2. Qualify with IBKR → get live/delayed pricing
      3. Place order at IBKR mid price
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

    # 1. AV fetch — select option contract (strike, expiration)
    logger.info(f"Selecting {option_type} for {symbol} ({action} signal, price=${signal_price:.2f})")
    option, quote, options_df = select_option_contract(symbol, action, signal_price)

    if option is None:
        reason = f"No suitable {option_type} option found for {symbol}"
        logger.warning(f"  REJECTED: {reason}")
        return {**base_result, 'status': 'failed', 'fill_price': 0.0,
                'filled_shares': 0, 'error': reason}

    strike = option['strike']
    expiration = option['expiration']
    av_bid = option.get('bid', 0) or 0
    av_ask = option.get('ask', 0) or 0
    av_mid = option.get('mid', 0) or ((av_bid + av_ask) / 2)
    oi = option.get('open_interest', 0) or 0
    vol = option.get('volume', 0) or 0
    iv = option.get('implied_volatility', 0) or 0
    delta = option.get('delta', 0) or 0

    base_result['strike'] = strike
    base_result['expiration'] = expiration
    base_result['premium'] = av_mid

    logger.info(f"  AV selection: {symbol} {expiration} {strike} {option_type.upper()}")
    logger.info(f"  AV Bid={av_bid:.2f} Ask={av_ask:.2f} Mid={av_mid:.2f} "
                f"OI={oi} Vol={vol} IV={iv:.2f} Delta={delta:.3f}")

    # 2. Validate guardrails on AV data (first-pass filter)
    rejection = validate_option(option, symbol)
    if rejection:
        logger.warning(f"  REJECTED: {rejection}")
        return {**base_result, 'status': 'failed', 'fill_price': 0.0,
                'filled_shares': 0, 'error': rejection}

    # 3. Dry run: log and return
    if dry_run:
        if raw_shares == 0:
            contracts = calculate_option_contracts(executor, av_mid)
        else:
            contracts = raw_shares
        logger.info(f"  [DRY RUN] Would BUY {contracts} x {symbol} {expiration} "
                     f"{strike} {option_type.upper()} @ ~${av_mid:.2f} "
                     f"(total ~${contracts * av_mid * 100:.2f})")
        return {**base_result, 'status': 'dry_run', 'fill_price': 0.0,
                'filled_shares': 0, 'error': ''}

    # 4. Qualify with IBKR
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

    # 5. Get IBKR live/delayed pricing (AV is end-of-day, likely stale)
    bid, ask, mid = av_bid, av_ask, av_mid
    ibkr_quote = get_ibkr_option_quote(executor.order_manager.ib, contract)
    if ibkr_quote:
        tag = "delayed" if ibkr_quote.get('delayed') else "live"
        logger.info(f"  IBKR {tag}: bid=${ibkr_quote['bid']:.2f} ask=${ibkr_quote['ask']:.2f} "
                    f"mid=${ibkr_quote['mid']:.2f} (AV was ${av_mid:.2f}, "
                    f"delta=${av_mid - ibkr_quote['mid']:+.2f})")
        bid = ibkr_quote['bid']
        ask = ibkr_quote['ask']
        mid = ibkr_quote['mid']
        base_result['premium'] = mid

        # Re-validate with live pricing
        live_option = {**option, 'bid': bid, 'ask': ask, 'mid': mid}
        rejection = validate_option(live_option, symbol)
        if rejection:
            logger.warning(f"  REJECTED (IBKR pricing): {rejection}")
            return {**base_result, 'status': 'failed', 'fill_price': 0.0,
                    'filled_shares': 0, 'error': rejection}
    else:
        logger.warning(f"  No IBKR quote — using AV data (mid=${av_mid:.2f})")

    # 6. Auto-size contracts with best available mid
    if raw_shares == 0:
        contracts = calculate_option_contracts(executor, mid)
        logger.info(f"  Sized: {contracts} contract(s) "
                     f"(${mid:.2f} x 100 = ${mid * 100:.2f}/contract)")
    else:
        contracts = raw_shares

    logger.info(f"  Final: {symbol} {expiration} {strike} {option_type.upper()} "
                f"x{contracts} @ ~${mid:.2f} (total ~${contracts * mid * 100:,.2f})")

    # 7. Create order — always BUY to open
    if use_market_orders:
        order = MarketOrder('BUY', contracts)
    else:
        limit_price = round(mid, 2)
        order = LimitOrder('BUY', contracts, limit_price)
        logger.info(f"  Limit order @ ${limit_price:.2f} (bid={bid:.2f}, ask={ask:.2f})")

    order.tif = 'DAY'
    acct = executor.order_manager.target_account
    if acct:
        order.account = acct

    # 8. Submit and wait for fill
    try:
        trade = executor.order_manager.ib.placeOrder(contract, order)
        order_id = trade.order.orderId
        logger.info(f"  Order submitted: BUY {contracts} x {symbol} {expiration} "
                     f"{strike}{right} (order_id={order_id})")

        executor.order_manager.ib.sleep(2)
        timeout = executor.order_manager.config.order_timeout
        start = time.time()
        repriced = False

        while time.time() - start < timeout:
            executor.order_manager.ib.sleep(0.5)
            if trade.isDone():
                break

        # If not filled, reprice closer to ask
        if not trade.isDone() and not use_market_orders and ask > bid:
            spread = ask - bid
            bump = round(spread * REPRICE_SPREAD_PCT, 2)
            new_limit = round(mid + bump, 2)
            logger.info(f"  Repricing: mid=${mid:.2f} + 15% spread (${spread:.2f}) "
                        f"= ${new_limit:.2f}")
            try:
                trade.order.lmtPrice = new_limit
                executor.order_manager.ib.placeOrder(contract, trade.order)
                repriced = True

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
            base_result['premium'] = fill_price
            return {**base_result, 'status': 'filled', 'fill_price': fill_price,
                    'filled_shares': filled, 'error': ''}
        else:
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
