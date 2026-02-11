"""
Real-Time Quote, Options, News/Sentiment & LLM Summary Utilities

Provides helper functions for fetching real-time stock quotes, options chain
snapshots, news sentiment, earnings data, and LLM-powered summarization
during market hours via the Alpha Vantage API and local ollama.

This is separate from the overnight batch data pull (pull_api_data.py).

Usage:
    from api_data.rt_utils import get_realtime_quote, get_options_snapshot
    from api_data.rt_utils import get_news_sentiment, get_earnings_overview, summarize_with_llm

    # Get current stock price
    quote = get_realtime_quote('AAPL')

    # Get options snapshot
    options = get_options_snapshot('AAPL', months_out=1, num_strikes=2)

    # Get news sentiment
    score, articles = get_news_sentiment('AAPL')

    # Get earnings + company overview
    fundamentals = get_earnings_overview('AAPL')

    # LLM summary (requires ollama + llama3.1:8b running locally)
    summary = summarize_with_llm('AAPL', articles, fundamentals)

CLI:
    python -m api_data.rt_utils AAPL                          # all sections (default)
    python -m api_data.rt_utils AAPL --news                   # news only
    python -m api_data.rt_utils AAPL --summary                # LLM summary (implies --news)
    python -m api_data.rt_utils AAPL --sector-analysis        # sector analysis only
    python -m api_data.rt_utils AAPL --data-path 'all_data_*.csv' --email  # full report emailed
"""

import argparse
import logging
import os
import smtplib
import time
from calendar import monthrange
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from api_data.collector import alpha_client
from stockformer.sector_features import SECTOR_ETF_MAP

logger = logging.getLogger(__name__)


def get_realtime_quotes_bulk(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch real-time quotes for up to 100 symbols in a single API call.

    Uses the REALTIME_BULK_QUOTES endpoint (premium).
    Returns full OHLCV per symbol: open, high, low, price (close), volume.

    Args:
        symbols: List of ticker symbols (max 100 per call;
                 larger lists are chunked automatically)

    Returns:
        DataFrame with columns: symbol, open, high, low, price, volume, timestamp
    """
    all_rows = []
    n_calls = (len(symbols) + 99) // 100
    is_sample = False

    # Chunk into batches of 100
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        symbol_str = ','.join(chunk)
        response = alpha_client.fetch(
            function='REALTIME_BULK_QUOTES', symbol=symbol_str)

        # Check for premium/sample data warning
        if 'message' in response and 'premium' in response.get('message', '').lower():
            is_sample = True

        # Parse response — API returns 'close' not 'price'
        if 'data' in response:
            for row in response['data']:
                # Use 'close' field (actual API), fall back to 'price' (legacy)
                price = float(row.get('close', 0) or row.get('price', 0))
                all_rows.append({
                    'symbol': row.get('symbol', ''),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'price': price,
                    'volume': int(row.get('volume', 0)),
                    'timestamp': row.get('timestamp', ''),
                })
        else:
            failed = list(response.keys())
            logger.warning(f"Bulk quote unexpected response keys: {failed}")
            print(f"  WARNING: Bulk quote returned unexpected response: {failed}")

    if not all_rows:
        return pd.DataFrame()

    if is_sample:
        print(f"  WARNING: Bulk endpoint returned SAMPLE data (premium key required)")

    df = pd.DataFrame(all_rows)
    print(f"  Bulk quotes: {len(df)} symbols fetched in {n_calls} API call(s)")
    return df


def get_realtime_quote(symbol: str) -> dict:
    """
    Fetch real-time stock quote from Alpha Vantage.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with keys: symbol, price, open, high, low, volume,
        prev_close, change, change_pct, latest_trading_day
    """
    response = alpha_client.fetch(function='GLOBAL_QUOTE', symbol=symbol)

    if 'Global Quote' not in response or not response['Global Quote']:
        raise ValueError(f"No quote data returned for {symbol}")

    gq = response['Global Quote']
    return {
        'symbol': gq.get('01. symbol', symbol),
        'price': float(gq.get('05. price', 0)),
        'open': float(gq.get('02. open', 0)),
        'high': float(gq.get('03. high', 0)),
        'low': float(gq.get('04. low', 0)),
        'volume': int(gq.get('06. volume', 0)),
        'prev_close': float(gq.get('08. previous close', 0)),
        'change': float(gq.get('09. change', 0)),
        'change_pct': gq.get('10. change percent', '0%').replace('%', ''),
        'latest_trading_day': gq.get('07. latest trading day', ''),
    }


def fetch_realtime_bollinger(symbol: str) -> pd.DataFrame:
    """
    Fetch live quote + BBANDS + RSI for bollinger strategy via Alpha Vantage API.

    Makes 3 API calls per symbol:
      1. GLOBAL_QUOTE → price, open, high, low, volume
      2. BBANDS (daily, close, period=20) → upper/middle/lower bands
      3. RSI (daily, close, period=14) → rsi_14

    Returns a 2-row DataFrame (yesterday + today) with columns matching
    the all_data CSV format used by bollinger_shadow_strategy.find_signals().

    Args:
        symbol: Stock ticker symbol

    Returns:
        pd.DataFrame with columns: date, symbol, adjusted_close, open, high, low,
        volume, bbands_upper_20, bbands_middle_20, bbands_lower_20, rsi_14
    """
    # 1. Real-time quote via GLOBAL_QUOTE endpoint
    quote = get_realtime_quote(symbol)

    # 2. BBANDS via Technical Analysis endpoint (daily, close, period=20)
    bbands_resp = alpha_client.fetch(
        function='BBANDS', symbol=symbol,
        interval='daily', series_type='close', time_period=20
    )
    bbands_key = 'Technical Analysis: BBANDS'
    if bbands_key not in bbands_resp:
        raise ValueError(f"No BBANDS data returned for {symbol}")
    bbands_df = pd.DataFrame.from_dict(bbands_resp[bbands_key], orient='index')
    bbands_df.index = pd.to_datetime(bbands_df.index)
    bbands_df = bbands_df.apply(pd.to_numeric).sort_index().tail(2)
    bbands_df = bbands_df.rename(columns={
        'Real Upper Band': 'bbands_upper_20',
        'Real Middle Band': 'bbands_middle_20',
        'Real Lower Band': 'bbands_lower_20',
    })

    # 3. RSI via Technical Analysis endpoint (daily, close, period=14)
    rsi_resp = alpha_client.fetch(
        function='RSI', symbol=symbol,
        interval='daily', series_type='close', time_period=14
    )
    rsi_key = 'Technical Analysis: RSI'
    if rsi_key not in rsi_resp:
        raise ValueError(f"No RSI data returned for {symbol}")
    rsi_df = pd.DataFrame.from_dict(rsi_resp[rsi_key], orient='index')
    rsi_df.index = pd.to_datetime(rsi_df.index)
    rsi_df = rsi_df.apply(pd.to_numeric).sort_index().tail(2)
    rsi_df = rsi_df.rename(columns={'RSI': 'rsi_14'})

    # Merge bbands + rsi on date index
    df = bbands_df.join(rsi_df, how='inner')
    df.index.name = 'date'
    df = df.reset_index()
    df['symbol'] = symbol

    # Map prices: yesterday's row gets prev_close, today's row gets live price.
    # This is critical for crossover detection (strategy compares row N-1 vs row N).
    if len(df) == 2:
        df.loc[df.index[0], 'adjusted_close'] = quote['prev_close']
        df.loc[df.index[1], 'adjusted_close'] = quote['price']
    else:
        df['adjusted_close'] = quote['price']

    df['open'] = quote['open']
    df['high'] = quote['high']
    df['low'] = quote['low']
    df['volume'] = quote['volume']

    return df


def fetch_realtime_batch(symbols: list[str], fetch_fn, timeout_per_symbol: int = 30) -> pd.DataFrame:
    """
    Fetch real-time data for multiple symbols using a per-symbol fetch function.

    Iterates symbols, calls fetch_fn(symbol) for each, skips failures with a
    warning, and concatenates results into a single DataFrame.

    Args:
        symbols: List of stock ticker symbols
        fetch_fn: Callable that takes a symbol string and returns a pd.DataFrame
        timeout_per_symbol: Max seconds per symbol (logged but not enforced via signal)

    Returns:
        pd.DataFrame with all symbols concatenated, or empty DataFrame if all fail
    """
    frames = []
    total = len(symbols)
    start = time.time()

    for i, sym in enumerate(symbols, 1):
        elapsed = time.time() - start
        rate = elapsed / i if i > 1 else 0
        eta = rate * (total - i)
        print(f"  Fetching {sym} ({i}/{total})... "
              f"[elapsed: {elapsed:.0f}s, est remaining: {eta:.0f}s]")
        try:
            df = fetch_fn(sym)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"    WARNING: {sym} failed: {e}")
            continue

    elapsed_total = time.time() - start
    succeeded = len(frames)
    failed = total - succeeded
    print(f"\n  Batch complete: {succeeded}/{total} symbols fetched "
          f"in {elapsed_total:.0f}s ({failed} failed)")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_third_friday(year: int, month: int) -> date:
    """
    Get the 3rd Friday of a given month/year (standard monthly options expiration).

    Args:
        year: Calendar year
        month: Calendar month (1-12)

    Returns:
        date object for the 3rd Friday
    """
    # Find the first day of the month
    first_day = date(year, month, 1)
    # Find the first Friday (weekday 4)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    # 3rd Friday = first Friday + 14 days
    return first_friday + timedelta(days=14)


def get_monthly_expiration(months_out: int = 1) -> date:
    """
    Compute the 3rd Friday of the month that is months_out from today.

    Args:
        months_out: Number of months forward (1 = next month, 0 = this month)

    Returns:
        date object for the target monthly expiration
    """
    today = date.today()
    # Calculate target month/year
    target_month = today.month + months_out
    target_year = today.year
    while target_month > 12:
        target_month -= 12
        target_year += 1

    target_exp = get_third_friday(target_year, target_month)

    # If months_out=0 and the 3rd Friday already passed, move to next month
    if months_out == 0 and target_exp < today:
        target_month += 1
        if target_month > 12:
            target_month = 1
            target_year += 1
        target_exp = get_third_friday(target_year, target_month)

    return target_exp


def find_closest_monthly_expiration(expirations: list[str], months_out: int = 1) -> str:
    """
    From a list of available expirations, find the one closest to the
    target monthly (3rd Friday) expiration.

    Args:
        expirations: List of expiration date strings (YYYY-MM-DD)
        months_out: Number of months forward

    Returns:
        The closest matching expiration string
    """
    target = get_monthly_expiration(months_out)
    target_dt = target

    # Find the expiration closest to our target 3rd Friday
    best = None
    best_diff = None
    for exp_str in expirations:
        exp_date = date.fromisoformat(exp_str)
        diff = abs((exp_date - target_dt).days)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = exp_str

    return best


def get_options_snapshot(
    symbol: str,
    months_out: int = 1,
    num_strikes: int = 2,
    option_type: str = 'both'
) -> tuple[dict, pd.DataFrame]:
    """
    Get a snapshot of options near the money for a given symbol.

    Fetches current stock price, then the latest options chain filtered to:
    - Monthly expiration closest to months_out target
    - ATM strike + num_strikes above and below
    - Calls, puts, or both

    Args:
        symbol: Stock ticker symbol
        months_out: Months out for expiration (1 = next month)
        num_strikes: Number of strikes above and below ATM
        option_type: 'call', 'put', or 'both'

    Returns:
        Tuple of (quote_dict, options_dataframe)
    """
    # Get current price
    quote = get_realtime_quote(symbol)
    current_price = quote['price']
    logger.info(f"{symbol} current price: ${current_price:.2f}")

    # Fetch latest options chain (no date = most recent)
    response = alpha_client.fetch(function='HISTORICAL_OPTIONS', symbol=symbol)

    if 'data' not in response or not response['data']:
        raise ValueError(f"No options data returned for {symbol}")

    df = pd.DataFrame(response['data'])

    # Convert numeric columns
    numeric_cols = ['strike', 'last', 'mark', 'bid', 'ask', 'bid_size', 'ask_size',
                    'volume', 'open_interest', 'implied_volatility',
                    'delta', 'gamma', 'theta', 'vega', 'rho']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Get available expirations and find closest monthly
    expirations = sorted(df['expiration'].unique())
    target_exp = find_closest_monthly_expiration(expirations, months_out)
    target_3rd_friday = get_monthly_expiration(months_out)
    logger.info(f"Target monthly exp (3rd Friday): {target_3rd_friday}, closest available: {target_exp}")

    # Filter to target expiration
    df = df[df['expiration'] == target_exp]

    # Filter by option type
    if option_type in ('call', 'put'):
        df = df[df['type'] == option_type]

    # Find ATM + nearby strikes
    all_strikes = sorted(df['strike'].unique())
    nearby_strikes = _get_nearby_strikes(all_strikes, current_price, num_strikes)

    # Filter to nearby strikes
    df = df[df['strike'].isin(nearby_strikes)]

    # Compute midpoint
    df = df.copy()
    df['mid'] = (df['bid'] + df['ask']) / 2

    # Select and order columns
    output_cols = ['strike', 'type', 'bid', 'ask', 'mid', 'last', 'volume',
                   'open_interest', 'implied_volatility', 'delta', 'gamma',
                   'theta', 'vega', 'expiration']
    available_cols = [c for c in output_cols if c in df.columns]
    df = df[available_cols].sort_values(['strike', 'type']).reset_index(drop=True)

    return quote, df


def _get_nearby_strikes(strike_prices: list[float], stock_price: float, num_strikes: int) -> list[float]:
    """
    Get ATM strike + num_strikes above and below.

    Args:
        strike_prices: Sorted list of available strikes
        stock_price: Current stock price
        num_strikes: Number of strikes on each side of ATM

    Returns:
        Sorted list of selected strikes
    """
    if not strike_prices:
        return []

    closest_idx = min(range(len(strike_prices)), key=lambda i: abs(strike_prices[i] - stock_price))
    start = max(0, closest_idx - num_strikes)
    end = min(len(strike_prices), closest_idx + num_strikes + 1)
    return strike_prices[start:end]


def get_news_sentiment(symbol: str, limit: int = 20) -> tuple[float, list[dict]]:
    """
    Fetch news sentiment for a symbol from Alpha Vantage NEWS_SENTIMENT endpoint.

    Computes weighted average: sum(relevance_score * ticker_sentiment_score) / sum(relevance_score)
    for the ticker-specific sentiment across the top articles.

    Args:
        symbol: Stock ticker symbol
        limit: Max number of articles to process (default 20)

    Returns:
        Tuple of (weighted_avg_score, articles_list) where each article dict has:
        title, source, summary, relevance_score, sentiment_score, weighted_score, time_published
    """
    response = alpha_client.fetch(function='NEWS_SENTIMENT', tickers=symbol)

    if 'feed' not in response or not response['feed']:
        logger.warning(f"No news data returned for {symbol}")
        return 0.0, []

    articles = []
    for item in response['feed'][:limit]:
        # Find this ticker's sentiment in the ticker_sentiment array
        ticker_entry = None
        for ts in item.get('ticker_sentiment', []):
            if ts.get('ticker', '').upper() == symbol.upper():
                ticker_entry = ts
                break

        if not ticker_entry:
            continue

        relevance = float(ticker_entry.get('relevance_score', 0))
        sentiment = float(ticker_entry.get('ticker_sentiment_score', 0))

        articles.append({
            'title': item.get('title', ''),
            'source': item.get('source', ''),
            'summary': item.get('summary', ''),
            'relevance_score': relevance,
            'sentiment_score': sentiment,
            'weighted_score': relevance * sentiment,
            'time_published': item.get('time_published', ''),
        })

    # Compute weighted average
    total_relevance = sum(a['relevance_score'] for a in articles)
    if total_relevance > 0:
        weighted_avg = sum(a['weighted_score'] for a in articles) / total_relevance
    else:
        weighted_avg = 0.0

    return weighted_avg, articles


def get_earnings_overview(symbol: str) -> dict:
    """
    Fetch earnings and company overview data from Alpha Vantage.

    Calls EARNINGS and OVERVIEW endpoints directly (no DB dependency).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with keys: symbol, sector, market_cap, forward_pe, 52w_high, 52w_low,
        analyst_ratings (dict), recent_quarters (list of last 2 quarters with EPS data)
    """
    result = {'symbol': symbol}

    # Fetch OVERVIEW
    try:
        overview = alpha_client.fetch(function='OVERVIEW', symbol=symbol)
        result['sector'] = overview.get('Sector', 'N/A')
        result['industry'] = overview.get('Industry', 'N/A')
        result['market_cap'] = overview.get('MarketCapitalization', 'N/A')
        result['forward_pe'] = overview.get('ForwardPE', 'N/A')
        result['52w_high'] = overview.get('52WeekHigh', 'N/A')
        result['52w_low'] = overview.get('52WeekLow', 'N/A')
        result['analyst_ratings'] = {
            'strong_buy': overview.get('AnalystRatingStrongBuy', '0'),
            'buy': overview.get('AnalystRatingBuy', '0'),
            'hold': overview.get('AnalystRatingHold', '0'),
            'sell': overview.get('AnalystRatingSell', '0'),
            'strong_sell': overview.get('AnalystRatingStrongSell', '0'),
        }
    except Exception as e:
        logger.warning(f"Failed to fetch OVERVIEW for {symbol}: {e}")

    # Fetch EARNINGS
    try:
        earnings = alpha_client.fetch(function='EARNINGS', symbol=symbol)
        quarterly = earnings.get('quarterlyEarnings', [])
        result['recent_quarters'] = []
        for q in quarterly[:2]:
            result['recent_quarters'].append({
                'fiscal_date': q.get('fiscalDateEnding', ''),
                'reported_eps': q.get('reportedEPS', 'N/A'),
                'estimated_eps': q.get('estimatedEPS', 'N/A'),
                'surprise_pct': q.get('surprisePercentage', 'N/A'),
            })
    except Exception as e:
        logger.warning(f"Failed to fetch EARNINGS for {symbol}: {e}")

    return result


def get_technical_data(symbol: str, data_path: str) -> dict | None:
    """
    Read the latest row of technical indicators for a symbol from a merged CSV.

    Args:
        symbol: Stock ticker symbol
        data_path: Path to CSV (supports globs like 'all_data_*.csv')

    Returns:
        Dict of technical indicator values, or None if symbol not found
    """
    import glob as glob_mod
    try:
        if '*' in data_path or '?' in data_path:
            files = sorted(glob_mod.glob(data_path))
            if not files:
                logger.warning(f"No files matching {data_path}")
                return None
            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(data_path)
    except Exception as e:
        logger.warning(f"Could not read data file {data_path}: {e}")
        return None

    sym_df = df[df['symbol'] == symbol].sort_values('date')
    if sym_df.empty:
        logger.warning(f"{symbol} not found in {data_path}")
        return None

    row = sym_df.iloc[-1]
    tech_cols = {
        # Moving averages
        'adjusted_close': 'Price',
        'sma_20': 'SMA 20', 'sma_50': 'SMA 50', 'sma_200': 'SMA 200',
        'ema_20': 'EMA 20', 'ema_50': 'EMA 50', 'ema_200': 'EMA 200',
        'sma_20_pct': 'Price vs SMA20 %', 'sma_50_pct': 'Price vs SMA50 %',
        'sma_200_pct': 'Price vs SMA200 %',
        # Oscillators
        'macd': 'MACD', 'macd_9_ema': 'MACD Signal Line',
        'rsi_14': 'RSI 14', 'adx_14': 'ADX 14', 'cci_14': 'CCI 14',
        # Volatility
        'atr_14': 'ATR 14',
        'bbands_upper_20': 'Bollinger Upper', 'bbands_middle_20': 'Bollinger Mid',
        'bbands_lower_20': 'Bollinger Lower',
        # Options
        'pcr': 'Put/Call Ratio',
        # Signals (-1/0/1)
        'macd_signal': 'MACD Signal', 'macd_zero_signal': 'MACD Zero Signal',
        'adx_signal': 'ADX Signal', 'atr_signal': 'ATR Signal',
        'bollinger_bands_signal': 'Bollinger Signal', 'rsi_signal': 'RSI Signal',
        'sma_cross_signal': 'SMA Cross Signal', 'cci_signal': 'CCI Signal',
        'pcr_signal': 'Put/Call Signal', 'bull_bear_delta': 'Bull/Bear Delta',
    }

    result = {'symbol': symbol, 'date': row.get('date', 'N/A')}
    for col, label in tech_cols.items():
        if col in row.index and pd.notna(row[col]):
            result[label] = row[col]

    return result


def summarize_technical_with_llm(symbol: str, tech_data: dict, model: str = 'llama3.1:8b') -> dict:
    """
    Use local ollama LLM to generate a 3-bullet technical outlook.

    Args:
        symbol: Stock ticker symbol
        tech_data: Dict from get_technical_data()
        model: Ollama model name

    Returns:
        Dict with 'bullets' (list[str], max 3, each under 150 chars)
    """
    try:
        import ollama
    except ImportError:
        logger.error("ollama package not installed. Run: pip install ollama")
        return {'bullets': []}

    # Build technical context
    lines = []
    for key, val in tech_data.items():
        if key in ('symbol', 'date'):
            continue
        lines.append(f"{key}: {val}")
    tech_ctx = '\n'.join(lines)

    prompt = f"""Given the following technical indicators for {symbol} as of {tech_data.get('date', 'latest')}, provide exactly 3 bullet points describing the technical outlook.

TECHNICAL INDICATORS:
{tech_ctx}

Rules:
- Each bullet must be under 150 characters
- Describe whether each signal is bullish, bearish, or neutral
- Reference specific indicator values (e.g. RSI, MACD, SMA crossovers)
- Be concise and direct

Format:
- [bullet 1]
- [bullet 2]
- [bullet 3]"""

    system_msg = "You are a technical analysis expert. Provide brief, precise technical outlook bullets. Each bullet must be under 150 characters. No disclaimers."

    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt},
        ])
        content = response['message']['content']
    except Exception as e:
        logger.error(f"LLM technical inference failed: {e}")
        return {'bullets': []}

    # Parse bullets (handle -, *, •, numbered)
    bullets = []
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Strip common bullet prefixes
        if line[0] in ('-', '*', '•'):
            bullet = line.lstrip('-*• ').strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] in ('.', ')'):
            bullet = line[2:].strip()
        else:
            continue
        if bullet:
            bullets.append(bullet[:150])

    return {'bullets': bullets[:3]}


def summarize_90day_with_llm(symbol: str, tech_data: dict, news_data: list[dict],
                              model: str = 'llama3.1:8b') -> dict:
    """
    Use local ollama LLM to generate a 3-bullet 90-day forward outlook.

    Combines technical indicators with high-relevance stock-specific news
    to produce medium-term outlook bullets (vs. the short-term technical outlook).

    Args:
        symbol: Stock ticker symbol
        tech_data: Dict from get_technical_data()
        news_data: List of article dicts from get_news_sentiment()
        model: Ollama model name

    Returns:
        Dict with 'bullets' (list[str], max 3, each under 150 chars)
    """
    try:
        import ollama
    except ImportError:
        logger.error("ollama package not installed. Run: pip install ollama")
        return {'bullets': []}

    # Build technical context
    lines = []
    for key, val in tech_data.items():
        if key in ('symbol', 'date'):
            continue
        lines.append(f"{key}: {val}")
    tech_ctx = '\n'.join(lines)

    # Build news context — only high-relevance (1.0) stock-specific articles
    high_rel = [a for a in (news_data or []) if a.get('relevance_score', 0) >= 0.99]
    news_ctx = ""
    for i, article in enumerate(high_rel[:10], 1):
        news_ctx += f"\n{i}. [{article['source']}] {article['title']}"
        if article['summary']:
            news_ctx += f"\n   {article['summary'][:200]}"
        news_ctx += f"\n   Sentiment: {article['sentiment_score']:.3f}"

    prompt = f"""Given {symbol}'s technical indicators and recent stock-specific news, provide exactly 3 bullet points about the 90-day forward outlook.

TECHNICAL INDICATORS (as of {tech_data.get('date', 'latest')}):
{tech_ctx}

STOCK-SPECIFIC NEWS ({len(high_rel)} high-relevance articles):
{news_ctx if news_ctx else "(no high-relevance articles)"}

Rules:
- Focus on MEDIUM-TERM (90-day) outlook, not day-to-day
- Each bullet must cite ONE specific data point: a number, price, or event
- Keep each bullet to ONE short sentence under 100 characters
- Be terse like a Bloomberg terminal alert
- No hedging, no disclaimers, no filler words

Format:
- [bullet 1]
- [bullet 2]
- [bullet 3]"""

    system_msg = "You are a stock analyst. Write terse, data-driven 90-day outlook bullets. One sentence per bullet, under 100 characters each. No disclaimers."

    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt},
        ])
        content = response['message']['content']
    except Exception as e:
        logger.error(f"LLM 90-day outlook failed: {e}")
        return {'bullets': []}

    # Parse bullets (handle -, *, •, numbered)
    bullets = []
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line[0] in ('-', '*', '•'):
            bullet = line.lstrip('-*• ').strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] in ('.', ')'):
            bullet = line[2:].strip()
        else:
            continue
        if bullet:
            bullets.append(bullet[:150])

    return {'bullets': bullets[:3]}


def summarize_with_llm(symbol: str, news_data: list[dict], earnings_data: dict, model: str = 'llama3.1:8b') -> dict:
    """
    Use local ollama LLM to summarize earnings + news into a condensed output.

    All inference runs locally on the GPU - no data leaves the machine.

    Args:
        symbol: Stock ticker symbol
        news_data: List of article dicts from get_news_sentiment()
        earnings_data: Dict from get_earnings_overview()
        model: Ollama model name (default: llama3.1:8b)

    Returns:
        Dict with 'summary' (str) and 'bullets' (list[str])
    """
    try:
        import ollama
    except ImportError:
        logger.error("ollama package not installed. Run: pip install ollama")
        return {'summary': 'Error: ollama not installed', 'bullets': []}

    # Build earnings context
    earnings_ctx = f"Company: {symbol}"
    if earnings_data.get('sector'):
        earnings_ctx += f" | Sector: {earnings_data.get('sector', 'N/A')}"
    if earnings_data.get('market_cap'):
        earnings_ctx += f" | Market Cap: {earnings_data.get('market_cap', 'N/A')}"
    if earnings_data.get('forward_pe'):
        earnings_ctx += f" | Forward P/E: {earnings_data.get('forward_pe', 'N/A')}"
    if earnings_data.get('52w_high'):
        earnings_ctx += f" | 52W High: {earnings_data.get('52w_high')} | 52W Low: {earnings_data.get('52w_low')}"

    ratings = earnings_data.get('analyst_ratings', {})
    if ratings:
        earnings_ctx += f"\nAnalyst Ratings - Strong Buy: {ratings.get('strong_buy', 0)}, Buy: {ratings.get('buy', 0)}, Hold: {ratings.get('hold', 0)}, Sell: {ratings.get('sell', 0)}, Strong Sell: {ratings.get('strong_sell', 0)}"

    quarters = earnings_data.get('recent_quarters', [])
    if quarters:
        earnings_ctx += "\nRecent Earnings:"
        for q in quarters:
            earnings_ctx += f"\n  {q['fiscal_date']}: Reported EPS={q['reported_eps']}, Estimated={q['estimated_eps']}, Surprise={q['surprise_pct']}%"

    # Build news context
    news_ctx = ""
    for i, article in enumerate(news_data[:20], 1):
        news_ctx += f"\n{i}. [{article['source']}] {article['title']}"
        if article['summary']:
            # Truncate long summaries
            summary = article['summary'][:200]
            news_ctx += f"\n   {summary}"
        news_ctx += f"\n   Sentiment: {article['sentiment_score']:.3f}, Relevance: {article['relevance_score']:.3f}"

    total_relevance = sum(a['relevance_score'] for a in news_data)
    if total_relevance > 0:
        weighted_avg = sum(a['weighted_score'] for a in news_data) / total_relevance
    else:
        weighted_avg = 0.0

    prompt = f"""Analyze {symbol}'s current market outlook based on the following data.

COMPANY CONTEXT:
{earnings_ctx}

RECENT NEWS (weighted sentiment score: {weighted_avg:.3f}):
{news_ctx}

Focus your analysis on:
- Current market sentiment and what recent news signals about near-term direction
- Key catalysts, risks, or events emerging from the news flow
- How sentiment aligns with or diverges from the fundamental picture
- Any sector or macro themes affecting the stock right now

Provide:
1. A concise summary paragraph (under 2500 characters) covering the current outlook, sentiment drivers, and near-term catalysts or risks.
2. Exactly 5 bullet points highlighting the most actionable or notable current developments.

Format your response as:
SUMMARY:
[your summary paragraph]

BULLETS:
- [bullet 1]
- [bullet 2]
- [bullet 3]
- [bullet 4]
- [bullet 5]"""

    system_msg = "You are a financial analyst assistant specializing in market sentiment and current outlook. Focus on what recent news and sentiment mean for near-term positioning. Be concise and data-driven. Do not add disclaimers."

    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt},
        ])
        content = response['message']['content']
    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        return {'summary': f'Error: {e}', 'bullets': []}

    # Parse response
    summary = ''
    bullets = []
    if 'SUMMARY:' in content:
        parts = content.split('BULLETS:')
        summary = parts[0].replace('SUMMARY:', '').strip()
        if len(parts) > 1:
            bullet_text = parts[1].strip()
            bullets = [b.strip().lstrip('- ') for b in bullet_text.split('\n') if b.strip().startswith('-')]
    else:
        summary = content.strip()

    return {'summary': summary, 'bullets': bullets[:5]}


def _build_earnings_context(symbol: str, earnings_data: dict) -> str:
    """Build a text block describing earnings & fundamentals for LLM prompts."""
    ctx = f"Company: {symbol}"
    if earnings_data.get('sector'):
        ctx += f" | Sector: {earnings_data.get('sector', 'N/A')}"
    if earnings_data.get('market_cap'):
        ctx += f" | Market Cap: {earnings_data.get('market_cap', 'N/A')}"
    if earnings_data.get('forward_pe'):
        ctx += f" | Forward P/E: {earnings_data.get('forward_pe', 'N/A')}"
    if earnings_data.get('52w_high'):
        ctx += f" | 52W High: {earnings_data.get('52w_high')} | 52W Low: {earnings_data.get('52w_low')}"

    ratings = earnings_data.get('analyst_ratings', {})
    if ratings:
        ctx += f"\nAnalyst Ratings - Strong Buy: {ratings.get('strong_buy', 0)}, Buy: {ratings.get('buy', 0)}, Hold: {ratings.get('hold', 0)}, Sell: {ratings.get('sell', 0)}, Strong Sell: {ratings.get('strong_sell', 0)}"

    quarters = earnings_data.get('recent_quarters', [])
    if quarters:
        ctx += "\nRecent Earnings:"
        for q in quarters:
            ctx += f"\n  {q['fiscal_date']}: Reported EPS={q['reported_eps']}, Estimated={q['estimated_eps']}, Surprise={q['surprise_pct']}%"

    return ctx


def summarize_earnings_with_llm(symbol: str, earnings_data: dict, model: str = 'llama3.1:8b') -> dict:
    """
    Use local ollama LLM to summarize earnings and significant figures only (no news).

    Args:
        symbol: Stock ticker symbol
        earnings_data: Dict from get_earnings_overview()
        model: Ollama model name (default: llama3.1:8b)

    Returns:
        Dict with 'summary' (str) and 'bullets' (list[str])
    """
    try:
        import ollama
    except ImportError:
        logger.error("ollama package not installed. Run: pip install ollama")
        return {'summary': 'Error: ollama not installed', 'bullets': []}

    earnings_ctx = _build_earnings_context(symbol, earnings_data)

    prompt = f"""Analyze {symbol} based on the following earnings and fundamental data.

EARNINGS & FUNDAMENTALS:
{earnings_ctx}

Focus your analysis on:
- Key business divisions and segments driving revenue and earnings
- Management guidance and forward-looking statements
- Earnings surprise direction and what it signals about business unit performance
- Analyst consensus shifts and what they imply about future quarters
- Any notable changes in margins, revenue mix, or segment contributions

Provide:
1. A concise summary paragraph (under 2500 characters) focused on divisional performance, management guidance, and forward outlook.
2. Exactly 5 bullet points highlighting the most important figures tied to business units, guidance, or forward-looking indicators.

Format your response as:
SUMMARY:
[your summary paragraph]

BULLETS:
- [bullet 1]
- [bullet 2]
- [bullet 3]
- [bullet 4]
- [bullet 5]"""

    system_msg = "You are a financial analyst assistant specializing in earnings analysis. Focus on business divisions, segment performance, management guidance, and forward-looking indicators. Be concise and data-driven. Do not add disclaimers."

    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt},
        ])
        content = response['message']['content']
    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        return {'summary': f'Error: {e}', 'bullets': []}

    # Parse response
    summary = ''
    bullets = []
    if 'SUMMARY:' in content:
        parts = content.split('BULLETS:')
        summary = parts[0].replace('SUMMARY:', '').strip()
        if len(parts) > 1:
            bullet_text = parts[1].strip()
            bullets = [b.strip().lstrip('- ') for b in bullet_text.split('\n') if b.strip().startswith('-')]
    else:
        summary = content.strip()

    return {'summary': summary, 'bullets': bullets[:5]}


def get_sector_etf(symbol: str, data_path: str = None) -> dict:
    """
    Map a stock symbol to its sector and corresponding sector ETF.

    Lookup order:
      1. all_data CSV (if data_path provided) — uses the 'sector' column
      2. Alpha Vantage OVERVIEW endpoint — uses the 'Sector' field

    Args:
        symbol: Stock ticker symbol
        data_path: Optional path to all_data CSV (supports globs)

    Returns:
        Dict with 'sector' (str), 'etf' (str or None), 'source' ('csv'|'api'|'unknown')
    """
    sector = None
    source = 'unknown'

    # Try CSV first (fast, no API call)
    if data_path:
        import glob as glob_mod
        try:
            if '*' in data_path or '?' in data_path:
                files = sorted(glob_mod.glob(data_path))
                if files:
                    # Read just the first file that has this symbol
                    for f in files:
                        df = pd.read_csv(f, usecols=['symbol', 'sector'])
                        match = df[df['symbol'] == symbol]
                        if not match.empty:
                            sector = match.iloc[-1]['sector']
                            source = 'csv'
                            break
            else:
                df = pd.read_csv(data_path, usecols=['symbol', 'sector'])
                match = df[df['symbol'] == symbol]
                if not match.empty:
                    sector = match.iloc[-1]['sector']
                    source = 'csv'
        except Exception as e:
            logger.warning(f"Could not read sector from data file: {e}")

    # Fallback to API
    if not sector or pd.isna(sector):
        try:
            overview = alpha_client.fetch(function='OVERVIEW', symbol=symbol)
            sector = overview.get('Sector', None)
            if sector:
                source = 'api'
        except Exception as e:
            logger.warning(f"Could not fetch sector from API for {symbol}: {e}")

    if not sector or pd.isna(sector):
        return {'sector': 'Unknown', 'etf': None, 'source': source}

    sector_upper = sector.upper().strip()
    etf = SECTOR_ETF_MAP.get(sector_upper)

    return {'sector': sector, 'etf': etf, 'source': source}


def summarize_sector_with_llm(sector: str, etf: str, articles: list[dict],
                               model: str = 'llama3.1:8b') -> dict:
    """
    Use local ollama LLM to generate a sector synopsis (<=500 chars).

    Args:
        sector: Sector name (e.g. "CONSUMER CYCLICAL")
        etf: Sector ETF symbol (e.g. "XLY")
        articles: News articles from get_news_sentiment(etf)
        model: Ollama model name

    Returns:
        Dict with 'synopsis' (str, <=500 chars) and 'etf' (str)
    """
    try:
        import ollama
    except ImportError:
        logger.error("ollama package not installed. Run: pip install ollama")
        return {'synopsis': 'Error: ollama not installed', 'etf': etf}

    # Build news context from sector ETF articles
    news_ctx = ""
    for i, article in enumerate(articles[:20], 1):
        news_ctx += f"\n{i}. [{article['source']}] {article['title']}"
        if article['summary']:
            news_ctx += f"\n   {article['summary'][:150]}"
        news_ctx += f"\n   Sentiment: {article['sentiment_score']:.3f}"

    total_relevance = sum(a['relevance_score'] for a in articles)
    if total_relevance > 0:
        weighted_avg = sum(a['weighted_score'] for a in articles) / total_relevance
    else:
        weighted_avg = 0.0

    prompt = f"""Analyze the {sector} sector as a whole (tracked by ETF: {etf}) based on the following recent news.

SECTOR NEWS (weighted sentiment: {weighted_avg:+.3f}):
{news_ctx}

RULES:
- Write about the SECTOR as a whole, NOT about individual stocks or companies by name
- Focus on broad sector-level themes: consumer spending, margins, regulation, macro trends
- Cover both the past 90 days and forward outlook for the next 90 days
- Your response MUST be a single complete paragraph of 3-4 sentences
- Your response MUST be under 450 characters total — count carefully
- End with a complete sentence — do NOT get cut off mid-thought
- No bullet points, no headers, no stock tickers"""

    system_msg = "You are a sector analyst. Write about sectors as a whole, never individual stocks. Keep responses under 450 characters. Always end with a complete sentence. No disclaimers."

    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt},
        ])
        synopsis = response['message']['content'].strip()
    except Exception as e:
        logger.error(f"LLM sector inference failed: {e}")
        return {'synopsis': f'Error: {e}', 'etf': etf}

    # If LLM still exceeded limit, trim to last complete sentence under 500 chars
    if len(synopsis) > 500:
        truncated = synopsis[:500]
        # Find last sentence-ending punctuation
        for end_char in ['. ', '! ', '? ']:
            last_period = truncated.rfind(end_char)
            if last_period > 200:
                synopsis = truncated[:last_period + 1]
                break
        else:
            # No sentence break found — just use the last period
            last_period = truncated.rfind('.')
            if last_period > 200:
                synopsis = truncated[:last_period + 1]

    return {'synopsis': synopsis, 'etf': etf, 'sector': sector, 'sentiment': weighted_avg}


def print_sector_analysis(symbol: str, data_path: str = None):
    """
    Fetch sector info, get sector ETF news, and print LLM sector synopsis.

    Args:
        symbol: Stock ticker symbol
        data_path: Optional path to all_data CSV

    Returns:
        Dict with sector analysis result, or None if sector unknown
    """
    sector_info = get_sector_etf(symbol, data_path)
    sector = sector_info['sector']
    etf = sector_info['etf']

    if not etf:
        print(f"\n  Could not map {symbol} (sector: {sector}) to a sector ETF.")
        return None

    print(f"\n{'=' * 70}")
    print(f"  BFT SECTOR ANALYSIS: {symbol}")
    print(f"  Sector: {sector}  |  ETF: {etf}")
    print(f"{'=' * 70}")
    print(f"  Fetching {etf} news sentiment...")

    _, articles = get_news_sentiment(etf, limit=20)
    if not articles:
        print(f"  No news found for sector ETF {etf}")
        return {'sector': sector, 'etf': etf, 'synopsis': 'No sector news available.', 'sentiment': 0.0}

    print(f"  {len(articles)} articles found. Generating sector synopsis...")
    result = summarize_sector_with_llm(sector, etf, articles)

    print(f"\n  {result['synopsis']}\n")
    print(f"  Sector Sentiment ({etf}): {result.get('sentiment', 0.0):+.4f}")
    print(f"{'=' * 70}\n")
    return result


def print_earnings_summary(symbol: str, earnings_data: dict = None):
    """Fetch earnings data (if not provided) and print LLM-generated earnings summary."""
    if earnings_data is None:
        earnings_data = get_earnings_overview(symbol)

    print(f"\n{'=' * 70}")
    print(f"  BFT AI EARNINGS REPORT SUMMARY: {symbol}")
    print(f"{'=' * 70}")
    print("  Generating earnings summary...")

    result = summarize_earnings_with_llm(symbol, earnings_data)

    print(f"\n  {result['summary']}\n")
    if result['bullets']:
        print(f"  Key Figures:")
        for b in result['bullets']:
            print(f"    - {b}")
    print(f"\n{'=' * 70}\n")
    return result


def print_news_sentiment(symbol: str, prefetched: tuple = None):
    """Print formatted news sentiment data."""
    if prefetched:
        weighted_avg, articles = prefetched
    else:
        weighted_avg, articles = get_news_sentiment(symbol)

    print(f"\n{'=' * 70}")
    print(f"  NEWS SENTIMENT: {symbol}")
    print(f"  Weighted Avg Score: {weighted_avg:+.4f}", end='')
    if weighted_avg > 0.15:
        print("  (Bullish)")
    elif weighted_avg < -0.15:
        print("  (Bearish)")
    else:
        print("  (Neutral)")
    print(f"  Articles analyzed: {len(articles)}")
    print(f"{'=' * 70}")

    display_limit = min(10, len(articles))
    for i, a in enumerate(articles[:10], 1):
        sentiment_label = 'Bullish' if a['sentiment_score'] > 0.15 else ('Bearish' if a['sentiment_score'] < -0.15 else 'Neutral')
        print(f"\n  {i}. {a['title'][:80]}")
        print(f"     Source: {a['source']}  |  {a['time_published'][:8] if a['time_published'] else ''}")
        print(f"     Sentiment: {a['sentiment_score']:+.3f} ({sentiment_label})  |  Relevance: {a['relevance_score']:.3f}")
    if len(articles) > 10:
        print(f"\n  ... {len(articles) - 10} more articles used for LLM analysis")

    print(f"\n{'=' * 70}\n")
    return weighted_avg, articles


def print_earnings_overview(symbol: str, earnings_data: dict):
    """Print formatted earnings and company overview data."""
    print(f"\n{'=' * 70}")
    print(f"  EARNINGS & OVERVIEW: {symbol}")
    print(f"{'=' * 70}")

    print(f"  Sector: {earnings_data.get('sector', 'N/A')}  |  Industry: {earnings_data.get('industry', 'N/A')}")

    # Format market cap
    mc = earnings_data.get('market_cap', 'N/A')
    if mc != 'N/A':
        try:
            mc_num = float(mc)
            if mc_num >= 1e12:
                mc = f"${mc_num/1e12:.2f}T"
            elif mc_num >= 1e9:
                mc = f"${mc_num/1e9:.2f}B"
            elif mc_num >= 1e6:
                mc = f"${mc_num/1e6:.2f}M"
        except (ValueError, TypeError):
            pass

    print(f"  Market Cap: {mc}  |  Forward P/E: {earnings_data.get('forward_pe', 'N/A')}")
    print(f"  52W High: {earnings_data.get('52w_high', 'N/A')}  |  52W Low: {earnings_data.get('52w_low', 'N/A')}")

    ratings = earnings_data.get('analyst_ratings', {})
    if ratings:
        print(f"\n  Analyst Ratings:")
        print(f"    Strong Buy: {ratings.get('strong_buy', 0)}  |  Buy: {ratings.get('buy', 0)}  |  "
              f"Hold: {ratings.get('hold', 0)}  |  Sell: {ratings.get('sell', 0)}  |  "
              f"Strong Sell: {ratings.get('strong_sell', 0)}")

    quarters = earnings_data.get('recent_quarters', [])
    if quarters:
        print(f"\n  Recent Quarterly Earnings:")
        print(f"  {'Quarter':>12}  {'Reported EPS':>13}  {'Estimated EPS':>14}  {'Surprise %':>11}")
        print(f"  {'-'*12}  {'-'*13}  {'-'*14}  {'-'*11}")
        for q in quarters:
            print(f"  {q['fiscal_date']:>12}  {str(q['reported_eps']):>13}  {str(q['estimated_eps']):>14}  {str(q['surprise_pct']):>10}%")

    print(f"{'=' * 70}\n")


def print_summary(symbol: str, news_data: list[dict], earnings_data: dict,
                  include_earnings_summary: bool = False,
                  tech_data: dict = None):
    """
    Print LLM-generated summary with optional multi-part layout.

    Sections (in order):
      1. BFT AI Current Outlook (news-driven, always)
      2. BFT Technical Outlook (technical indicators, if tech_data provided)
      3. BFT 90 Day Outlook (technicals + stock-specific news, if both available)
      4. BFT Latest Earnings Summary (divisions/guidance, if include_earnings_summary)

    Returns:
        Tuple of (outlook_result, earnings_summary_result, technical_result, outlook_90d_result)
    """
    if include_earnings_summary:
        print(f"\n{'=' * 70}")
        print(f"  BFT AI SUMMARY AND EARNINGS REPORT: {symbol}")
        print(f"{'=' * 70}")

    # Part 1: Current Outlook (news-driven)
    print(f"\n  {'─' * 50}")
    print(f"  BFT AI CURRENT OUTLOOK: {symbol}")
    print(f"  {'─' * 50}")
    print("  Generating current outlook...")

    outlook_result = summarize_with_llm(symbol, news_data, earnings_data)

    print(f"\n  {outlook_result['summary']}\n")
    if outlook_result['bullets']:
        print(f"  Key Points:")
        for b in outlook_result['bullets']:
            print(f"    - {b}")

    # Part 2: Technical Outlook (from merged data)
    technical_result = None
    if tech_data:
        print(f"\n  {'─' * 50}")
        print(f"  BFT TECHNICAL OUTLOOK: {symbol}")
        print(f"  {'─' * 50}")
        print(f"  (data as of {tech_data.get('date', 'N/A')})")

        technical_result = summarize_technical_with_llm(symbol, tech_data)

        if technical_result['bullets']:
            for b in technical_result['bullets']:
                print(f"    - {b}")

    # Part 3: 90 Day Outlook (technicals + stock-specific news)
    outlook_90d_result = None
    if tech_data and news_data:
        print(f"\n  {'─' * 50}")
        print(f"  BFT 90 DAY OUTLOOK: {symbol}")
        print(f"  {'─' * 50}")

        outlook_90d_result = summarize_90day_with_llm(symbol, tech_data, news_data)

        if outlook_90d_result['bullets']:
            for b in outlook_90d_result['bullets']:
                print(f"    - {b}")

    # Part 4: Earnings Summary (divisions/guidance/forward-looking)
    earnings_summary_result = None
    if include_earnings_summary:
        print(f"\n  {'─' * 50}")
        print(f"  BFT LATEST EARNINGS SUMMARY: {symbol}")
        print(f"  {'─' * 50}")
        print("  Generating earnings summary...")

        earnings_summary_result = summarize_earnings_with_llm(symbol, earnings_data)

        print(f"\n  {earnings_summary_result['summary']}\n")
        if earnings_summary_result['bullets']:
            print(f"  Key Figures:")
            for b in earnings_summary_result['bullets']:
                print(f"    - {b}")

    print(f"\n{'=' * 70}\n")
    return outlook_result, earnings_summary_result, technical_result, outlook_90d_result


def print_snapshot(symbol: str, months_out: int = 1, num_strikes: int = 2, option_type: str = 'both'):
    """Print a formatted stock quote + options snapshot to stdout."""
    quote, options = get_options_snapshot(symbol, months_out, num_strikes, option_type)
    _print_snapshot_from_data(quote, options, months_out, num_strikes, option_type)


def _print_snapshot_from_data(quote: dict, options: pd.DataFrame, months_out: int = 1, num_strikes: int = 2, option_type: str = 'both'):
    """Print a formatted stock quote + options snapshot from pre-fetched data."""
    # Stock quote
    change_sign = '+' if quote['change'] >= 0 else ''
    print(f"\n{'=' * 70}")
    print(f"  {quote['symbol']}  ${quote['price']:.2f}  "
          f"{change_sign}{quote['change']:.2f} ({change_sign}{quote['change_pct']}%)")
    print(f"  Open: ${quote['open']:.2f}  High: ${quote['high']:.2f}  "
          f"Low: ${quote['low']:.2f}  Vol: {quote['volume']:,}")
    print(f"  As of: {quote['latest_trading_day']}")
    print(f"{'=' * 70}")

    if options.empty:
        print("\n  No options data found for the specified criteria.\n")
        return

    exp = options['expiration'].iloc[0]
    target = get_monthly_expiration(months_out)
    print(f"  Expiration: {exp}  (target 3rd Friday: {target})")
    print(f"  Showing ATM +/- {num_strikes} strikes | Type: {option_type}")
    print(f"{'-' * 70}")

    # Separate calls and puts
    calls = options[options['type'] == 'call'] if 'type' in options.columns else pd.DataFrame()
    puts = options[options['type'] == 'put'] if 'type' in options.columns else pd.DataFrame()

    atm_strike = min(options['strike'].unique(), key=lambda s: abs(s - quote['price']))

    if not calls.empty:
        print(f"\n  CALLS:")
        print(f"  {'Strike':>8}  {'Bid':>8}  {'Ask':>8}  {'Mid':>8}  {'Last':>8}  {'Vol':>6}  {'OI':>7}  {'IV':>7}  {'Delta':>7}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
        for _, row in calls.iterrows():
            marker = ' *' if row['strike'] == atm_strike else '  '
            print(f"{marker}{row['strike']:>8.2f}  {row['bid']:>8.2f}  {row['ask']:>8.2f}  "
                  f"{row['mid']:>8.2f}  {row['last']:>8.2f}  {row.get('volume', 0):>6.0f}  "
                  f"{row.get('open_interest', 0):>7.0f}  {row.get('implied_volatility', 0):>7.3f}  "
                  f"{row.get('delta', 0):>7.4f}")

    if not puts.empty:
        print(f"\n  PUTS:")
        print(f"  {'Strike':>8}  {'Bid':>8}  {'Ask':>8}  {'Mid':>8}  {'Last':>8}  {'Vol':>6}  {'OI':>7}  {'IV':>7}  {'Delta':>7}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
        for _, row in puts.iterrows():
            marker = ' *' if row['strike'] == atm_strike else '  '
            print(f"{marker}{row['strike']:>8.2f}  {row['bid']:>8.2f}  {row['ask']:>8.2f}  "
                  f"{row['mid']:>8.2f}  {row['last']:>8.2f}  {row.get('volume', 0):>6.0f}  "
                  f"{row.get('open_interest', 0):>7.0f}  {row.get('implied_volatility', 0):>7.3f}  "
                  f"{row.get('delta', 0):>7.4f}")

    print(f"\n  * = ATM strike (closest to ${quote['price']:.2f})")
    print(f"{'=' * 70}\n")


def build_email_html(symbol: str, quote: dict, options: pd.DataFrame,
                     news_data: list[dict] = None, weighted_avg: float = None,
                     earnings_data: dict = None, llm_result: dict = None,
                     earnings_summary: dict = None,
                     technical_result: dict = None,
                     sector_result: dict = None,
                     outlook_90d_result: dict = None) -> str:
    """
    Build an HTML email body with quote, options, news, earnings, LLM summary,
    sector analysis, and 90-day outlook sections.

    Args:
        symbol: Stock ticker symbol
        quote: Quote dict from get_realtime_quote()
        options: Options DataFrame from get_options_snapshot()
        news_data: List of article dicts (optional)
        weighted_avg: Weighted sentiment score (optional)
        earnings_data: Earnings/overview dict (optional)
        llm_result: Current outlook LLM summary dict (optional)
        earnings_summary: Earnings-only LLM summary dict (optional)
        technical_result: Technical outlook LLM dict (optional)
        sector_result: Sector analysis dict from print_sector_analysis() (optional)
        outlook_90d_result: 90-day outlook LLM dict (optional)

    Returns:
        HTML string
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    change_sign = '+' if quote['change'] >= 0 else ''
    change_color = '#2e7d32' if quote['change'] >= 0 else '#c62828'

    html = f"""
    <html><body style="font-family: 'Courier New', monospace; background: #f5f5f5; padding: 20px;">
    <div style="max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px;">

    <h2 style="border-bottom: 2px solid #333; padding-bottom: 8px;">
      {symbol} &nbsp; ${quote['price']:.2f} &nbsp;
      <span style="color: {change_color};">{change_sign}{quote['change']:.2f} ({change_sign}{quote['change_pct']}%)</span>
    </h2>
    <p style="color: #666; font-size: 12px;">Report generated: {timestamp} | Data as of: {quote['latest_trading_day']}</p>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 16px;">
      <tr>
        <td style="padding: 4px 12px;"><b>Open:</b> ${quote['open']:.2f}</td>
        <td style="padding: 4px 12px;"><b>High:</b> ${quote['high']:.2f}</td>
        <td style="padding: 4px 12px;"><b>Low:</b> ${quote['low']:.2f}</td>
        <td style="padding: 4px 12px;"><b>Volume:</b> {quote['volume']:,}</td>
      </tr>
    </table>
    """

    # Options table
    if not options.empty:
        exp = options['expiration'].iloc[0]
        atm_strike = min(options['strike'].unique(), key=lambda s: abs(s - quote['price']))

        html += f"""
        <h3 style="border-bottom: 1px solid #ccc; padding-bottom: 4px;">Options Chain &mdash; Exp: {exp}</h3>
        """
        for opt_type in ['call', 'put']:
            subset = options[options['type'] == opt_type] if 'type' in options.columns else pd.DataFrame()
            if subset.empty:
                continue
            html += f"<h4>{opt_type.upper()}S</h4>"
            html += """<table style="border-collapse: collapse; width: 100%; font-size: 13px;">
            <tr style="background: #e0e0e0;">
              <th style="padding: 4px 6px; text-align: right;">Strike</th>
              <th style="padding: 4px 6px; text-align: right;">Bid</th>
              <th style="padding: 4px 6px; text-align: right;">Ask</th>
              <th style="padding: 4px 6px; text-align: right;">Mid</th>
              <th style="padding: 4px 6px; text-align: right;">Last</th>
              <th style="padding: 4px 6px; text-align: right;">Vol</th>
              <th style="padding: 4px 6px; text-align: right;">OI</th>
              <th style="padding: 4px 6px; text-align: right;">IV</th>
              <th style="padding: 4px 6px; text-align: right;">Delta</th>
            </tr>"""
            for _, row in subset.iterrows():
                bg = '#fff9c4' if row['strike'] == atm_strike else 'white'
                html += f"""<tr style="background: {bg};">
                  <td style="padding: 3px 6px; text-align: right;">{row['strike']:.2f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row['bid']:.2f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row['ask']:.2f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row['mid']:.2f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row['last']:.2f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row.get('volume', 0):.0f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row.get('open_interest', 0):.0f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row.get('implied_volatility', 0):.3f}</td>
                  <td style="padding: 3px 6px; text-align: right;">{row.get('delta', 0):.4f}</td>
                </tr>"""
            html += "</table>"

    # Earnings & Overview
    if earnings_data and earnings_data.get('sector'):
        mc = earnings_data.get('market_cap', 'N/A')
        if mc != 'N/A':
            try:
                mc_num = float(mc)
                if mc_num >= 1e12:
                    mc = f"${mc_num/1e12:.2f}T"
                elif mc_num >= 1e9:
                    mc = f"${mc_num/1e9:.2f}B"
                elif mc_num >= 1e6:
                    mc = f"${mc_num/1e6:.2f}M"
            except (ValueError, TypeError):
                pass

        html += f"""
        <h3 style="border-bottom: 1px solid #ccc; padding-bottom: 4px;">Earnings &amp; Overview</h3>
        <table style="border-collapse: collapse; margin-bottom: 8px;">
          <tr><td style="padding: 2px 10px;"><b>Sector:</b></td><td>{earnings_data.get('sector', 'N/A')}</td>
              <td style="padding: 2px 10px;"><b>Industry:</b></td><td>{earnings_data.get('industry', 'N/A')}</td></tr>
          <tr><td style="padding: 2px 10px;"><b>Market Cap:</b></td><td>{mc}</td>
              <td style="padding: 2px 10px;"><b>Forward P/E:</b></td><td>{earnings_data.get('forward_pe', 'N/A')}</td></tr>
          <tr><td style="padding: 2px 10px;"><b>52W High:</b></td><td>{earnings_data.get('52w_high', 'N/A')}</td>
              <td style="padding: 2px 10px;"><b>52W Low:</b></td><td>{earnings_data.get('52w_low', 'N/A')}</td></tr>
        </table>
        """
        ratings = earnings_data.get('analyst_ratings', {})
        if ratings:
            html += "<p><b>Analyst Ratings:</b> "
            html += f"Strong Buy: {ratings.get('strong_buy', 0)} | Buy: {ratings.get('buy', 0)} | "
            html += f"Hold: {ratings.get('hold', 0)} | Sell: {ratings.get('sell', 0)} | "
            html += f"Strong Sell: {ratings.get('strong_sell', 0)}</p>"

        quarters = earnings_data.get('recent_quarters', [])
        if quarters:
            html += """<table style="border-collapse: collapse; font-size: 13px; margin-bottom: 12px;">
            <tr style="background: #e0e0e0;">
              <th style="padding: 4px 8px;">Quarter</th>
              <th style="padding: 4px 8px;">Reported EPS</th>
              <th style="padding: 4px 8px;">Estimated EPS</th>
              <th style="padding: 4px 8px;">Surprise %</th>
            </tr>"""
            for q in quarters:
                html += f"""<tr>
                  <td style="padding: 3px 8px;">{q['fiscal_date']}</td>
                  <td style="padding: 3px 8px; text-align: right;">{q['reported_eps']}</td>
                  <td style="padding: 3px 8px; text-align: right;">{q['estimated_eps']}</td>
                  <td style="padding: 3px 8px; text-align: right;">{q['surprise_pct']}%</td>
                </tr>"""
            html += "</table>"

    # BFT AI Summary — two-part layout when both are available
    has_outlook = llm_result and llm_result.get('summary')
    has_earnings_llm = earnings_summary and earnings_summary.get('summary')

    if has_outlook and has_earnings_llm:
        # Umbrella heading
        html += f"""<h3 style="border-bottom: 2px solid #333; padding-bottom: 4px;">BFT AI Summary and Earnings Report</h3>"""

    # Part 1: Current Outlook
    if has_outlook:
        html += f"""
        <h4 style="border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #1565c0;">BFT AI Current Outlook</h4>
        <p style="line-height: 1.5;">{llm_result['summary']}</p>
        """
        if llm_result.get('bullets'):
            html += "<ul>"
            for b in llm_result['bullets']:
                html += f"<li>{b}</li>"
            html += "</ul>"

    # Technical Outlook
    if technical_result and technical_result.get('bullets'):
        html += f"""
        <h4 style="border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #6a1b9a;">BFT Technical Outlook</h4>
        <ul>"""
        for b in technical_result['bullets']:
            html += f"<li>{b}</li>"
        html += "</ul>"

    # 90 Day Outlook
    if outlook_90d_result and outlook_90d_result.get('bullets'):
        html += f"""
        <h4 style="border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #00695c;">BFT 90 Day Outlook</h4>
        <ul>"""
        for b in outlook_90d_result['bullets']:
            html += f"<li>{b}</li>"
        html += "</ul>"

    # Sector Analysis
    if sector_result and sector_result.get('synopsis'):
        sect_sentiment = sector_result.get('sentiment', 0.0)
        if sect_sentiment > 0.15:
            sect_label, sect_color = 'Bullish', '#2e7d32'
        elif sect_sentiment < -0.15:
            sect_label, sect_color = 'Bearish', '#c62828'
        else:
            sect_label, sect_color = 'Neutral', '#f57f17'

        html += f"""
        <h3 style="border-bottom: 1px solid #ccc; padding-bottom: 4px;">BFT Sector Analysis</h3>
        <p><b>Sector:</b> {sector_result.get('sector', 'N/A')} &nbsp; | &nbsp;
           <b>ETF:</b> {sector_result.get('etf', 'N/A')} &nbsp; | &nbsp;
           <b>Sentiment:</b> <span style="color: {sect_color}; font-weight: bold;">{sect_sentiment:+.4f} ({sect_label})</span></p>
        <p style="line-height: 1.5; background: #f9f9f9; padding: 10px; border-left: 3px solid {sect_color};">
            {sector_result['synopsis']}
        </p>
        """

    # News Sentiment
    if news_data is not None and weighted_avg is not None:
        if weighted_avg > 0.15:
            sent_label, sent_color = 'Bullish', '#2e7d32'
        elif weighted_avg < -0.15:
            sent_label, sent_color = 'Bearish', '#c62828'
        else:
            sent_label, sent_color = 'Neutral', '#f57f17'

        html += f"""
        <h3 style="border-bottom: 1px solid #ccc; padding-bottom: 4px;">News Sentiment</h3>
        <p><b>Weighted Avg Score:</b>
          <span style="color: {sent_color}; font-weight: bold;">{weighted_avg:+.4f} ({sent_label})</span>
          &nbsp; | &nbsp; Articles: {len(news_data)}</p>
        <table style="border-collapse: collapse; width: 100%; font-size: 12px;">
        """
        for i, a in enumerate(news_data[:10], 1):
            a_color = '#2e7d32' if a['sentiment_score'] > 0.15 else ('#c62828' if a['sentiment_score'] < -0.15 else '#555')
            html += f"""<tr style="border-bottom: 1px solid #eee;">
              <td style="padding: 4px; vertical-align: top; width: 20px;">{i}.</td>
              <td style="padding: 4px;">
                <b>{a['title'][:100]}</b><br>
                <span style="color: #888;">{a['source']} | {a['time_published'][:8] if a['time_published'] else ''}</span>
                &nbsp; <span style="color: {a_color};">Sentiment: {a['sentiment_score']:+.3f}</span>
                &nbsp; Relevance: {a['relevance_score']:.3f}
              </td>
            </tr>"""
        html += "</table>"

    # Part 2: Earnings Summary
    if has_earnings_llm:
        html += f"""
        <h4 style="border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #2e7d32;">BFT Latest Earnings Summary</h4>
        <p style="line-height: 1.5;">{earnings_summary['summary']}</p>
        """
        if earnings_summary.get('bullets'):
            html += "<ul>"
            for b in earnings_summary['bullets']:
                html += f"<li>{b}</li>"
            html += "</ul>"

    html += """
    <hr style="margin-top: 20px;">
    <p style="color: #999; font-size: 11px;">Generated by bar_fly_trading rt_utils</p>
    </div></body></html>
    """
    return html


def send_email_report(subject: str, html_body: str) -> bool:
    """
    Send an HTML email report using SMTP credentials from environment variables.

    Uses the same env vars as ibkr/notifier.py:
        IBKR_SMTP_SERVER, IBKR_SMTP_PORT, IBKR_SMTP_USER,
        IBKR_SMTP_PASSWORD, IBKR_NOTIFY_EMAIL (comma-separated recipients)

    Args:
        subject: Email subject line
        html_body: HTML content for the email body

    Returns:
        True if sent successfully, False otherwise
    """
    smtp_server = os.getenv('IBKR_SMTP_SERVER')
    smtp_port = int(os.getenv('IBKR_SMTP_PORT', '587'))
    smtp_user = os.getenv('IBKR_SMTP_USER')
    smtp_password = os.getenv('IBKR_SMTP_PASSWORD')
    notify_email = os.getenv('IBKR_NOTIFY_EMAIL')

    if not all([smtp_server, smtp_user, smtp_password, notify_email]):
        logger.error("Email not configured. Set IBKR_SMTP_SERVER, IBKR_SMTP_USER, IBKR_SMTP_PASSWORD, IBKR_NOTIFY_EMAIL env vars.")
        print("  ERROR: Email not configured. Required env vars:")
        print("    IBKR_SMTP_SERVER, IBKR_SMTP_USER, IBKR_SMTP_PASSWORD, IBKR_NOTIFY_EMAIL")
        return False

    recipients = [e.strip() for e in notify_email.split(',')]

    try:
        sent = []
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            for recipient in recipients:
                msg = MIMEMultipart('alternative')
                msg['From'] = smtp_user
                msg['To'] = recipient
                msg['Subject'] = subject
                msg.attach(MIMEText(html_body, 'html'))
                server.send_message(msg)
                sent.append(recipient)

        print(f"  Email sent individually to {len(sent)} recipient(s): {', '.join(sent)}")
        logger.info(f"Email report sent: {subject} -> {sent}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        if sent:
            print(f"  Partial send ({len(sent)}/{len(recipients)}): {', '.join(sent)}")
        print(f"  ERROR: Failed to send email: {e}")
        return False


def check_ollama_available() -> bool:
    """Check if ollama Python package is installed and the server is reachable."""
    try:
        import ollama
        ollama.list()
        return True
    except ImportError:
        return False
    except Exception:
        # Package installed but server not running
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real-time stock quote, options, news sentiment & LLM summary',
    )
    parser.add_argument('symbol', type=str, help='Stock ticker symbol')
    parser.add_argument('--months-out', type=int, default=1,
                        help='Months out for expiration (default: 1)')
    parser.add_argument('--strikes', type=int, default=2,
                        help='Number of strikes above/below ATM (default: 2)')
    parser.add_argument('--type', type=str, default='both', choices=['call', 'put', 'both'],
                        help='Option type filter (default: both)')
    parser.add_argument('--news', action='store_true',
                        help='Fetch and display news sentiment')
    parser.add_argument('--summary', action='store_true',
                        help='Run LLM summary of news + earnings (implies --news, requires ollama)')
    parser.add_argument('--earnings-summary', action='store_true',
                        help='Run LLM summary of earnings & fundamentals only (requires ollama)')
    parser.add_argument('--sector-analysis', action='store_true',
                        help='Run sector ETF news analysis with LLM synopsis (requires ollama)')
    parser.add_argument('--no-earnings', action='store_true',
                        help='Skip earnings data in --summary (news-only LLM summary)')
    parser.add_argument('--email', action='store_true',
                        help='Send the output as an HTML email report')
    parser.add_argument('--email-to', type=str, default=None,
                        help='Override IBKR_NOTIFY_EMAIL and send only to this address')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to all_data CSV for technical indicators and sector lookup')

    args = parser.parse_args()

    # Default-all: if no section flags given, enable all sections
    section_flags = [args.news, args.summary, args.earnings_summary, args.sector_analysis]
    if not any(section_flags):
        args.news = True
        args.summary = True
        args.earnings_summary = True
        args.sector_analysis = True

    # Pre-flight: verify ollama is available if any LLM sections requested
    needs_ollama = args.summary or args.earnings_summary or args.sector_analysis
    ollama_ok = False
    if needs_ollama:
        ollama_ok = check_ollama_available()
        if not ollama_ok:
            print("  ERROR: ollama is not available (package missing or server not running).")
            print("  LLM sections (summary, earnings-summary, sector-analysis) will be skipped.")
            if args.email:
                print("  Email will NOT be sent without LLM content.")
            args.summary = False
            args.earnings_summary = False
            args.sector_analysis = False

    # Always show the quote + options snapshot
    try:
        quote, options = get_options_snapshot(args.symbol, args.months_out, args.strikes, args.type)
    except Exception as e:
        logger.error(f"Failed to fetch options for {args.symbol}: {e}")
        # Fall back to quote-only
        try:
            quote = get_realtime_quote(args.symbol)
            options = pd.DataFrame()
            logger.info(f"Continuing with quote only (no options data)")
        except Exception as e2:
            logger.error(f"Failed to fetch quote for {args.symbol}: {e2}")
            print(f"Error: Could not fetch data for {args.symbol}. Check symbol and API key.")
            import sys
            sys.exit(1)
    # Print snapshot using already-fetched data
    _print_snapshot_from_data(quote, options, args.months_out, args.strikes, args.type)

    # --summary implies --news
    if args.summary:
        args.news = True

    # Track data for email
    news_data = None
    weighted_avg = None
    earnings_data = None
    llm_result = None
    earnings_summary_result = None
    technical_result = None
    outlook_90d_result = None
    tech_data = None
    sector_result = None

    # Fetch technical data if data-path provided
    if args.data_path:
        tech_data = get_technical_data(args.symbol, args.data_path)
        if tech_data:
            print(f"  Technical data loaded for {args.symbol} (as of {tech_data.get('date', 'N/A')})")
        else:
            print(f"  WARNING: No technical data found for {args.symbol} in {args.data_path}")

    # 2) Earnings & Overview (raw data)
    needs_earnings = (args.summary and not args.no_earnings) or args.earnings_summary
    if needs_earnings:
        earnings_data = get_earnings_overview(args.symbol)
        print_earnings_overview(args.symbol, earnings_data)
    elif args.summary and args.no_earnings:
        earnings_data = {'symbol': args.symbol}

    # Fetch news data silently (needed for LLM before display)
    if args.news:
        weighted_avg, news_data = get_news_sentiment(args.symbol)

    # 3) BFT AI Summary — two-part when earnings included, optional technical
    if args.summary:
        include_earnings = needs_earnings
        try:
            llm_result, earnings_summary_result, technical_result, outlook_90d_result = print_summary(
                args.symbol, news_data, earnings_data,
                include_earnings_summary=include_earnings,
                tech_data=tech_data,
            )
        except Exception as e:
            logger.error(f"LLM summary failed: {e}")
            print(f"\n  [BFT AI Summary unavailable: {e}]\n")

    # 4) News Sentiment (display after LLM summary)
    if args.news:
        print_news_sentiment(args.symbol, prefetched=(weighted_avg, news_data))

    # 5) BFT AI Earnings Report Summary (standalone, only when --earnings-summary without --summary)
    if args.earnings_summary and not args.summary:
        try:
            earnings_summary_result = print_earnings_summary(args.symbol, earnings_data)
        except Exception as e:
            logger.error(f"LLM earnings summary failed: {e}")
            print(f"\n  [BFT AI Earnings Summary unavailable: {e}]\n")

    # 6) BFT Sector Analysis
    if args.sector_analysis:
        try:
            sector_result = print_sector_analysis(args.symbol, args.data_path)
        except Exception as e:
            logger.error(f"Sector analysis failed: {e}")
            print(f"\n  [BFT Sector Analysis unavailable: {e}]\n")

    # Send email if requested
    if args.email and needs_ollama and not ollama_ok:
        print("  Skipping email — ollama was required but not available.")
    elif args.email:
        # Override recipient if --email-to is set
        if args.email_to:
            os.environ['IBKR_NOTIFY_EMAIL'] = args.email_to
        change_sign = '+' if quote['change'] >= 0 else ''
        subject = f"{args.symbol} ${quote['price']:.2f} {change_sign}{quote['change']:.2f} ({change_sign}{quote['change_pct']}%) - RT Report"
        html = build_email_html(
            symbol=args.symbol,
            quote=quote,
            options=options,
            news_data=news_data,
            weighted_avg=weighted_avg,
            earnings_data=earnings_data,
            llm_result=llm_result,
            earnings_summary=earnings_summary_result,
            technical_result=technical_result,
            sector_result=sector_result,
            outlook_90d_result=outlook_90d_result,
        )
        send_email_report(subject, html)