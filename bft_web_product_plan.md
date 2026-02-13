# BFT Web Product Plan

## Vision

A research-focused stock dashboard organized by S&P 500 sectors. Users see 11 sector cards + SPY/QQQ, color-coded red/green by real-time performance. Tapping a sector drills into its constituent stocks. Tapping a stock flips the card to reveal an LLM-generated technical + news summary. Mobile-first, swipeable, eventually audio — think "TikTok for stock research."

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         User's Browser / PWA      │
                    │                                    │
                    │  ┌──────────────────────────────┐  │
                    │  │  13 Sector Cards (red/green)  │  │
                    │  │  SPY  QQQ  XLF  XLK  XLV ... │  │
                    │  └──────────┬───────────────────┘  │
                    │             │ tap                   │
                    │  ┌──────────▼───────────────────┐  │
                    │  │  Stock Grid (e.g. XLF)        │  │
                    │  │  JPM  BAC  WFC  GS  MS  ...   │  │
                    │  └──────────┬───────────────────┘  │
                    │             │ tap / swipe           │
                    │  ┌──────────▼───────────────────┐  │
                    │  │  Card Flip → Full Report      │  │
                    │  │  Technical Outlook             │  │
                    │  │  AI News Summary               │  │
                    │  │  Bollinger / Signal Status     │  │
                    │  │  [🔊 Listen]                   │  │
                    │  └──────────────────────────────┘  │
                    └──────────────┬─────────────────────┘
                                  │
                    HTTPS + WebSocket (wss://)
                                  │
                    ┌──────────────▼─────────────────────┐
                    │          EC2 (t3.small)             │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │  Nginx (reverse proxy + SSL) │    │
                    │  │  yourdomain.com → Next.js    │    │
                    │  │  api.yourdomain.com → FastAPI│    │
                    │  └─────────┬───────────────────┘    │
                    │            │                         │
                    │  ┌─────────▼───────────────────┐    │
                    │  │  FastAPI Backend              │    │
                    │  │                               │    │
                    │  │  GET /api/sectors             │    │
                    │  │  GET /api/sector/{xlf}        │    │
                    │  │  GET /api/symbol/{JPM}        │    │
                    │  │  GET /api/signals/today       │    │
                    │  │  WS  /ws/prices (live)        │    │
                    │  └─────────┬───────────────────┘    │
                    │            │                         │
                    │  ┌─────────▼───────────────────┐    │
                    │  │  Data Layer (JSON on disk)    │    │
                    │  │                               │    │
                    │  │  /var/www/bft/data/           │    │
                    │  │  ├── sectors.json             │    │
                    │  │  ├── sector_XLF.json          │    │
                    │  │  ├── sector_XLK.json          │    │
                    │  │  ├── ...                      │    │
                    │  │  ├── AAPL.json                │    │
                    │  │  ├── JPM.json                 │    │
                    │  │  └── signals_today.json       │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │  Cron Jobs                    │    │
                    │  │                               │    │
                    │  │  Every 5 min (market hours):  │    │
                    │  │    update_quotes.py            │    │
                    │  │    → yfinance bulk quote       │    │
                    │  │    → writes sectors.json       │    │
                    │  │    → pushes via WebSocket      │    │
                    │  │                               │    │
                    │  │  Daily 6am ET:                 │    │
                    │  │    generate_reports.py          │    │
                    │  │    → Alpha Vantage technicals  │    │
                    │  │    → Ollama LLM summaries      │    │
                    │  │    → writes {SYMBOL}.json      │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │  Existing bar_fly_trading     │    │
                    │  │  (unchanged)                  │    │
                    │  │                               │    │
                    │  │  rt_utils.py                  │    │
                    │  │  ├── get_realtime_quote()     │    │
                    │  │  ├── get_technical_data()     │    │
                    │  │  ├── summarize_with_llm()     │    │
                    │  │  ├── summarize_technical_..() │    │
                    │  │  └── get_news_sentiment()     │    │
                    │  │                               │    │
                    │  │  bollinger_shadow_strategy.py  │    │
                    │  │  └── find_signals()            │    │
                    │  └─────────────────────────────┘    │
                    └─────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  External APIs                  │
                    │  ├── Alpha Vantage (technicals) │
                    │  ├── yfinance (bulk quotes)     │
                    │  └── Ollama (local LLM)         │
                    └────────────────────────────────┘
```

## Data Flow

```
Market Open (9:30am ET)
│
├── Every 5 min ──────────────────────────────────────────────┐
│   update_quotes.py                                          │
│   1. yfinance.download(spy_500_symbols, period='1d')        │
│   2. Compute per-sector avg change (XLF = mean of JPM,BAC…) │
│   3. Write sectors.json + sector_{XLF}.json                 │
│   4. Push deltas via WebSocket → browser updates cards      │
│   Cost: $0 (yfinance is free, ~2 sec for 500 symbols)      │
└─────────────────────────────────────────────────────────────┘

Daily 6:00am ET ──────────────────────────────────────────────┐
│   generate_reports.py                                       │
│   For each of 500 SPY symbols:                              │
│   1. get_technical_data(sym, 'all_data_*.csv')              │
│   2. summarize_technical_with_llm(sym, tech_data)           │
│   3. get_news_sentiment(sym)                                │
│   4. summarize_with_llm(sym, news, earnings)                │
│   5. Write {SYMBOL}.json                                    │
│                                                             │
│   Rate budget: 500 × 3 AV calls = 1500 calls               │
│   At 150 req/min (premium) = ~10 min                        │
│   LLM inference: 500 × ~5 sec = ~42 min (local Ollama)     │
│   Total: ~1 hour daily                                      │
└─────────────────────────────────────────────────────────────┘
```

## User Experience

### Level 1: Sector Overview
```
┌─────────────────────────────────────────────────┐
│  BFT Research          Feb 6, 2026  3:42pm ET   │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌───────┐  ┌───────┐                           │
│  │  SPY  │  │  QQQ  │                           │
│  │ ▲1.2% │  │ ▲0.8% │                           │
│  └───────┘  └───────┘                           │
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │ XLF │ │ XLK │ │ XLV │ │ XLE │ │ XLY │      │
│  │▲1.3%│ │▲0.9%│ │▼0.4%│ │▲2.1%│ │▼0.2%│      │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │ XLI │ │ XLB │ │ XLC │ │ XLU │ │XLRE │      │
│  │▲0.5%│ │▼0.1%│ │▲1.1%│ │▲0.3%│ │▼0.7%│      │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
│                                                  │
│  Signals Today: 3 BUY, 1 SELL (Bollinger)       │
└─────────────────────────────────────────────────┘

Cards: green fill = up, red fill = down
       intensity scales with magnitude (▲2.1% is deeper green)
```

### Level 2: Sector Drill-Down (tap XLF)
```
┌─────────────────────────────────────────────────┐
│  ← Financials (XLF ▲1.3%)                       │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │ JPM │ │ BAC │ │ WFC │ │ GS  │ │ MS  │      │
│  │▲2.1%│ │▲1.8%│ │▼0.3%│ │▲1.5%│ │▲0.9%│      │
│  │$247 │ │$41  │ │$72  │ │$598 │ │$128 │      │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │ BLK │ │ SCHW│ │ AXP │ │ CB  │ │ CME │      │
│  │▲0.7%│ │▲1.2%│ │▲0.4%│ │▼0.5%│ │▲0.3%│      │
│  │$1042│ │$85  │ │$295 │ │$278 │ │$245 │      │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
│  ...                                             │
│                                                  │
│  Sort: [Change %] [Price] [RSI] [Signal]         │
└─────────────────────────────────────────────────┘

Mobile: swipe right to go back to sectors
```

### Level 3: Stock Card Flip (tap JPM)
```
┌─────────────────────────────────────────────────┐
│  JPM $247.32 ▲2.1% (+$5.09)            [flip]  │
├─────────────────────────────────────────────────┤
│                                                  │
│  BFT Technical Outlook (as of Feb 6)             │
│  ─────────────────────────────────────           │
│  Bullish crossover: EMA20 above SMA20,           │
│  indicating potential uptrend.                   │
│                                                  │
│  - RSI 58.3: neutral, below overbought (70)     │
│  - MACD bearish divergence (-0.47)               │
│  - Bollinger: trading near middle band           │
│  - ADX 22.1: weak trend strength                 │
│                                                  │
│  BFT AI Current Outlook                          │
│  ─────────────────────────────────────           │
│  Strong Q4 earnings beat driven by investment    │
│  banking revenue recovery. Consumer credit       │
│  quality remains stable.                         │
│                                                  │
│  - Net interest income up 4% YoY                 │
│  - Trading revenue exceeded estimates            │
│  - Management raised 2026 NII guidance           │
│                                                  │
│  Signal: No active Bollinger signal              │
│                                                  │
│  [🔊 Listen]  [📧 Email Report]  [📌 Watch]     │
└─────────────────────────────────────────────────┘

Mobile: swipe left/right for next/prev stock in sector
        swipe down to go back to grid
```

## Phased Build Plan

### Phase 1: Working Prototype (1-2 weeks)

**Goal:** Sectors + stocks + card flip with daily reports. No live updates yet.

| Task | Effort | Details |
|------|--------|---------|
| `web/generate_reports.py` | 3 hrs | Wraps existing rt_utils, outputs JSON per symbol |
| `web/build_sector_map.py` | 1 hr | Parse all_data CSVs for sector column, group symbols, write sectors.json |
| `web/api.py` (FastAPI) | 2 hrs | 5 endpoints, reads JSON from disk |
| Next.js frontend: sector grid | 4 hrs | 13 cards, color-coded, responsive grid |
| Next.js frontend: stock grid | 4 hrs | Per-sector stock cards with sorting |
| Next.js frontend: card flip | 4 hrs | CSS 3D flip animation, report display |
| Nginx + SSL + DNS | 1 hr | Certbot, A record for domain |
| Cron: daily report generation | 30 min | 6am ET trigger |
| **Total** | **~20 hrs** | |

**Cost:** $0 incremental (EC2 already running, domain already owned).

### Phase 2: Real-Time + Mobile UX (2-4 weeks)

**Goal:** Live red/green updates, swipe navigation, PWA installable on phone.

| Task | Effort | Details |
|------|--------|---------|
| `web/update_quotes.py` | 2 hrs | yfinance bulk download every 5 min, write JSON |
| WebSocket server (FastAPI) | 3 hrs | Push price deltas to connected clients |
| WebSocket client (Next.js) | 3 hrs | Update card colors without page refresh |
| Swipe gestures (Framer Motion) | 4 hrs | Left/right between stocks, up/down for drill/back |
| PWA manifest + service worker | 2 hrs | Installable on phone home screen |
| Text-to-speech | 2 hrs | Browser Web Speech API, "Listen" button on card |
| Cron: 5-min quote updates | 30 min | Market hours only (9:30-4:00 ET, M-F) |
| **Total** | **~17 hrs** | |

**Cost:** $0 incremental. yfinance is free. Web Speech API is free.

### Phase 3: Multi-User + Auth (2-3 weeks)

**Goal:** User accounts, personal watchlists, free/paid tiers.

| Task | Effort | Details |
|------|--------|---------|
| Supabase setup (auth + DB) | 2 hrs | Email/Google login, user table |
| Next.js auth integration | 4 hrs | Login/signup pages, session management |
| Watchlist feature | 4 hrs | Save/load per user, pin symbols |
| Email digest opt-in | 2 hrs | Daily email with your existing notifier |
| Free vs. paid tier logic | 3 hrs | Free = sectors + top 5 per sector; paid = full |
| Rate limiting middleware | 2 hrs | Per-user API limits |
| **Total** | **~17 hrs** | |

**New cost:** Supabase free tier (50k monthly active users) → $25/mo if exceeded.

### Phase 4: Scale + Monetize (when traction exists)

| Task | Details | Cost |
|------|---------|------|
| Move frontend to Vercel | CDN, auto-scaling, preview deploys | Free → $20/mo |
| CloudFront in front of API | Cache JSON responses | ~$5/mo |
| Stripe subscriptions | $10/mo paid tier | 2.9% + $0.30/txn |
| Upgrade data provider | Polygon.io or similar for reliability | $30-100/mo |
| PostgreSQL (Supabase) | Historical reports, user analytics | $25/mo |
| Push notifications | PWA push for signal alerts | $0 (Web Push API) |

**Break-even:** ~$100/mo costs → 10 paid subscribers.

## Cost Summary

| Phase | Monthly Cost | What's New |
|-------|-------------|------------|
| Phase 1 | ~$15 | EC2 (already have) + domain (already own) |
| Phase 2 | ~$15 | Same (yfinance + Web Speech are free) |
| Phase 3 | ~$15-40 | + Supabase ($0-25 depending on users) |
| Phase 4 | ~$75-160 | + Vercel + CloudFront + Polygon + Supabase |

## Tech Stack Summary

```
Frontend:  Next.js 14 (App Router)
           Tailwind CSS (styling)
           Framer Motion (animations, swipe gestures)
           WebSocket client (live prices)
           Web Speech API (text-to-speech)
           PWA (installable)

Backend:   FastAPI (Python — same language as bar_fly_trading)
           WebSocket server (fastapi-websockets)
           JSON file storage (Phase 1-2) → PostgreSQL (Phase 3+)

Data:      yfinance (free real-time quotes)
           Alpha Vantage (technicals, news, earnings)
           Ollama (local LLM for summaries)
           Existing bar_fly_trading functions (unchanged)

Infra:     EC2 t3.small (already running)
           Nginx + Let's Encrypt SSL
           Cron (systemd timers)
           → Phase 4: Vercel (frontend) + CloudFront (API cache)
```

## API Endpoints

```
GET  /api/sectors
     → { sectors: [{id: "XLF", name: "Financials", change_pct: 1.3, ...}, ...],
         indices: [{id: "SPY", price: 523.45, change_pct: 1.2}, ...] }

GET  /api/sector/{sector_id}
     → { sector: "XLF", name: "Financials", change_pct: 1.3,
         stocks: [{symbol: "JPM", price: 247.32, change_pct: 2.1, rsi: 58.3, signal: null}, ...] }

GET  /api/symbol/{symbol}
     → { symbol: "JPM", price: 247.32, change_pct: 2.1,
         technical: { ema_sma_cross: "bullish", rsi: 58.3, macd_diff: -0.47, bullets: [...] },
         news: { sentiment: 0.72, summary: "...", bullets: [...] },
         earnings: { ... },
         signal: { type: null, bollinger_position: "mid-band" },
         report_date: "2026-02-06" }

GET  /api/signals/today
     → { signals: [{symbol: "AMZN", type: "BUY", strategy: "bollinger", reason: "..."}, ...] }

WS   /ws/prices
     → Stream: { "JPM": {"price": 247.45, "change_pct": 2.15}, ... }
        (delta updates every 5 min during market hours)
```

## JSON File Structure

```
/var/www/bft/data/
├── sectors.json              # 13 entries (11 sectors + SPY + QQQ)
├── sector_XLF.json           # All financials in SPY with current prices
├── sector_XLK.json           # All tech in SPY
├── ...                       # One per sector
├── AAPL.json                 # Full report: technical + news + earnings + signals
├── JPM.json
├── ...                       # One per SPY constituent (~500 files)
├── signals_today.json        # Today's bollinger/strategy signals
└── meta.json                 # Last update timestamps, symbol count, etc.
```

## SPY Sector Mapping (11 GICS Sectors)

| ETF  | Sector | ~Stocks in SPY |
|------|--------|----------------|
| XLF  | Financials | ~70 |
| XLK  | Technology | ~65 |
| XLV  | Health Care | ~65 |
| XLE  | Energy | ~25 |
| XLY  | Consumer Discretionary | ~50 |
| XLI  | Industrials | ~80 |
| XLB  | Materials | ~25 |
| XLC  | Communication Services | ~25 |
| XLU  | Utilities | ~30 |
| XLRE | Real Estate | ~30 |
| XLP  | Consumer Staples | ~35 |

Total: ~500 stocks. Sector assignments already exist in your `all_data_*.csv` files (`sector` column).

## Future Ideas

- **Swipe-to-trade:** Swipe right on a signal card → place 1-share paper trade via IBKR
- **Historical accuracy tracker:** For each past signal, did price actually revert? Show win rate
- **Sector rotation heatmap:** 30-day sector performance as a calendar heatmap
- **Earnings calendar:** Upcoming earnings dates highlighted on stock cards
- **Custom screener:** User defines criteria (RSI < 30 AND price > SMA200), see matching stocks
- **Social/sharing:** Share a card view as an image (canvas screenshot → clipboard)
- **Dark mode:** Toggle, stored in user preferences
- **Audio feed:** Auto-play summaries like a podcast — "Here's your market morning brief"
- **Model predictions overlay:** Show stockformer 3d/10d/30d predictions on stock cards
- **Comparison mode:** Side-by-side two stocks in same sector