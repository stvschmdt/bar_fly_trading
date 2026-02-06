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
    python -m api_data.rt_utils AAPL
    python -m api_data.rt_utils AAPL --news
    python -m api_data.rt_utils AAPL --news --summary
    python -m api_data.rt_utils AAPL --months-out 2 --strikes 3 --news --summary
"""

import argparse
import logging
import os
import smtplib
from calendar import monthrange
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from api_data.collector import alpha_client

logger = logging.getLogger(__name__)


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
                  include_earnings_summary: bool = False):
    """
    Print LLM-generated summary with optional two-part layout.

    When include_earnings_summary=True, prints two subsections:
      1. BFT AI Current Outlook (news-driven)
      2. BFT Latest Earnings Summary (divisions/guidance/forward-looking)

    When False, prints a single "BFT AI Current Outlook" section.

    Returns:
        Tuple of (outlook_result, earnings_summary_result) — earnings_summary_result
        is None if include_earnings_summary is False.
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

    # Part 2: Earnings Summary (divisions/guidance/forward-looking)
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
    return outlook_result, earnings_summary_result


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
                     earnings_summary: dict = None) -> str:
    """
    Build an HTML email body with quote, options, news, earnings, and LLM summary sections.

    When both llm_result and earnings_summary are provided, renders two subsections:
      - BFT AI Current Outlook (news-driven)
      - BFT Latest Earnings Summary (divisions/guidance/forward-looking)

    Args:
        symbol: Stock ticker symbol
        quote: Quote dict from get_realtime_quote()
        options: Options DataFrame from get_options_snapshot()
        news_data: List of article dicts (optional)
        weighted_avg: Weighted sentiment score (optional)
        earnings_data: Earnings/overview dict (optional)
        llm_result: Current outlook LLM summary dict (optional)
        earnings_summary: Earnings-only LLM summary dict (optional)

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
    parser.add_argument('--no-earnings', action='store_true',
                        help='Skip earnings data in --summary (news-only LLM summary)')
    parser.add_argument('--email', action='store_true',
                        help='Send the output as an HTML email report')
    parser.add_argument('--email-to', type=str, default=None,
                        help='Override IBKR_NOTIFY_EMAIL and send only to this address')

    args = parser.parse_args()

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

    # 3) BFT AI Summary — two-part when earnings included
    if args.summary:
        include_earnings = needs_earnings
        try:
            llm_result, earnings_summary_result = print_summary(
                args.symbol, news_data, earnings_data,
                include_earnings_summary=include_earnings,
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

    # Send email if requested
    if args.email:
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
        )
        send_email_report(subject, html)