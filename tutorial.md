## bar_fly_trading - End to End Tutorial

### 1. Setup

#### 1.1 Database

Option A: Docker MySQL
```bash
docker pull mysql:latest
docker run --name mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -e MYSQL_DATABASE=bar_fly_trading -p 3306:3306 -d mysql:latest
```

Option B: Native MySQL
```bash
# Ubuntu/Debian
sudo apt install mysql-server
sudo systemctl start mysql
sudo mysql -u root -e "CREATE DATABASE bar_fly_trading; ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'my-secret-pw'; FLUSH PRIVILEGES;"

# macOS
brew install mysql && brew services start mysql
mysql -u root -e "CREATE DATABASE bar_fly_trading; ALTER USER 'root'@'localhost' IDENTIFIED BY 'my-secret-pw'; FLUSH PRIVILEGES;"
```

Option C: Clone an existing database from another server
```bash
# On source server
mysqldump -u root -p bar_fly_trading > bar_fly_trading_dump.sql

# Copy and import locally
scp user@source-server:bar_fly_trading_dump.sql .
mysql -u root -pmy-secret-pw bar_fly_trading < bar_fly_trading_dump.sql
```

#### 1.2 Environment Variables

```bash
export MYSQL_PASSWORD=my-secret-pw
export ALPHAVANTAGE_API_KEY=your_key_here
```

For email notifications (optional):
```bash
export IBKR_SMTP_SERVER=smtp.gmail.com
export IBKR_SMTP_USER=you@gmail.com
export IBKR_SMTP_PASSWORD=your_app_password
export IBKR_NOTIFY_EMAIL=recipient1@gmail.com,recipient2@gmail.com
```

#### 1.3 Dependencies

```bash
pip install -r requirements.txt
```

For LLM summaries (optional):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
pip install ollama
```

---

### 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                      │
│                                                                                 │
│  ┌─────────────────────┐          ┌─────────────────────────────────────────┐   │
│  │   Alpha Vantage API  │          │          IBKR Gateway (AWS EC2)         │   │
│  │   (external, paid)   │          │          (separate machine)             │   │
│  │                      │          │                                         │   │
│  │  - OHLCV prices      │          │  Port 4001 (live) / 4002 (paper)       │   │
│  │  - Technical indic.  │          │  Requires SSH tunnel to access:        │   │
│  │  - Fundamentals      │          │  ssh -L 4001:127.0.0.1:4001 user@aws  │   │
│  │  - Options chains    │          │                                         │   │
│  │  - Economic data     │          │  - Account balances                    │   │
│  │  - News sentiment    │          │  - Portfolio positions                 │   │
│  │  - Earnings          │          │  - Order placement/execution           │   │
│  │  - Company overview  │          │  - Real-time market data               │   │
│  └──────────┬───────────┘          └──────────────────┬──────────────────────┘   │
│             │                                         │                          │
└─────────────┼─────────────────────────────────────────┼──────────────────────────┘
              │ REST API                                │ IB API (TCP)
              │                                         │
┌─────────────┼─────────────────────────────────────────┼──────────────────────────┐
│             │           LOCAL MACHINE                  │                          │
│             ▼                                         │                          │
│  ┌─────────────────────┐                              │                          │
│  │   api_data/          │                              │                          │
│  │   pull_api_data      │──────────┐                   │                          │
│  │   rt_utils           │          │                   │                          │
│  │   backfill_options   │          │                   │                          │
│  └─────────────────────┘          │                   │                          │
│                                    ▼                   │                          │
│  ┌─────────────────────┐   ┌──────────────┐           │                          │
│  │    cron.py           │──>│  MySQL DB    │           │                          │
│  │  (nightly pipeline)  │   │              │           │                          │
│  │  1. pull_api_data    │   │ core_stock   │           │                          │
│  │  2. screener         │   │ options      │           │                          │
│  │  3. overnight PDF    │   │ fundamentals │           │                          │
│  │  4. Google Drive     │   │ gold_table   │           │                          │
│  └─────────────────────┘   └──────┬───────┘           │                          │
│                                    │                   │                          │
│                                    ▼                   │                          │
│                            ┌──────────────┐           │                          │
│                            │ all_data_*.csv│           │                          │
│                            └──────┬───────┘           │                          │
│                                    │                   │                          │
│           ┌────────────────────────┤                   │                          │
│           │                        │                   │                          │
│           ▼                        ▼                   │                          │
│  ┌─────────────────┐   ┌─────────────────────┐        │                          │
│  │ visualizations/  │   │ stockformer/         │        │                          │
│  │ screener         │   │ main (train + infer) │        │                          │
│  │                  │   │ 9 models:            │        │                          │
│  │ - tech charts    │   │  3 horizons x        │        │                          │
│  │ - overnight PDF  │   │  3 label modes       │        │                          │
│  └─────────────────┘   └──────────┬───────────┘        │                          │
│                                    │                   │                          │
│                                    ▼                   │                          │
│                         ┌─────────────────────┐        │                          │
│                         │ merge_predictions    │        │                          │
│                         └──────────┬──────────┘        │                          │
│                                    │                   │                          │
│                                    ▼                   │                          │
│                         ┌─────────────────────┐        │                          │
│                         │merged_predictions.csv│        │                          │
│                         └──────────┬──────────┘        │                          │
│                                    │                   │                          │
│           ┌────────────────────────┼──────────┐        │                          │
│           │                        │          │        │                          │
│           ▼                        ▼          ▼        │                          │
│  ┌────────────────┐  ┌──────────────┐  ┌───────────┐  │                          │
│  │ portfolio.py    │  │ run_backtest │  │run_regr.  │  │                          │
│  │ (filter/rank)   │  │ (ML strat)   │  │momentum   │  │                          │
│  │                 │  │              │  │           │  │                          │
│  │ --price-above   │  │ pred_class=1 │  │ pred > 1% │  │                          │
│  │ --top-k-sharpe  │  │ pred_class=0 │  │ adx > 0   │  │                          │
│  │ --filter-field  │  │              │  │ cci < 0   │  │                          │
│  └───────┬────────┘  └──────┬───────┘  └─────┬─────┘  │                          │
│          │                  │                │         │                          │
│          ▼                  ▼                ▼         │                          │
│  ┌────────────────┐  ┌────────────────────────────┐   │                          │
│  │portfolio_picks  │  │ backtest_stats             │   │                          │
│  │.csv             │  │ - per-symbol P&L           │   │                          │
│  │(symbol list,    │  │ - win/loss, win rate       │   │                          │
│  │ pipeable via    │  │ - Sharpe ratio             │   │                          │
│  │ --symbols-file) │  │ - trades.csv               │   │                          │
│  └───────┬────────┘  │ - pending_orders.csv        │   │                          │
│          │           └──────────┬─────────────────┘   │                          │
│          │                      │                      │                          │
│          └──────────┬───────────┘                      │                          │
│                     │                                  │                          │
│                     ▼                                  │                          │
│  ┌──────────────────────────────────────────┐          │                          │
│  │          signals/pending_orders.csv       │          │                          │
│  │  action,symbol,shares,price,strategy,... │          │                          │
│  └──────────────────┬───────────────────────┘          │                          │
│                     │                                  │                          │
│                     ▼                                  │                          │
│  ┌──────────────────────────────────────────┐          │                          │
│  │         ibkr/execute_signals             │          │                          │
│  │                                          │          │                          │
│  │  --dry-run  (preview, no connection)     │          │                          │
│  │  (default)  (paper trading)          ────┼──────────┘                          │
│  │  --live     (real money)             ────┼── requires trading_mode.conf=live   │
│  │                                          │                                    │
│  │  Modes:                                  │                                    │
│  │  --signals file.csv  (one-shot)          │                                    │
│  │  --watch             (poll directory)    │                                    │
│  │                                          │                                    │
│  │         ibkr/trading_mode.conf           │                                    │
│  │         ┌───────────────────┐            │                                    │
│  │         │ TRADING_MODE=paper│ ← default  │                                    │
│  │         │ TRADING_MODE=live │ ← manual   │                                    │
│  │         └───────────────────┘            │                                    │
│  └──────────────────┬───────────────────────┘                                    │
│                     │                                                            │
│                     ▼                                                            │
│  ┌──────────────────────────────────────────┐                                    │
│  │       signals/executed/ (archive)         │                                    │
│  │  - original signal CSV (timestamped)      │                                    │
│  │  - execution results CSV (fills, errors)  │                                    │
│  └──────────────────────────────────────────┘                                    │
│                                                                                  │
│  ┌──────────────────────────────────────────┐                                    │
│  │       template_strategy.py               │                                    │
│  │  Copy + customize for new strategies     │                                    │
│  │                                          │                                    │
│  │  CUSTOMIZE:                              │                                    │
│  │    _check_entry_conditions()             │                                    │
│  │    _check_exit_custom()                  │                                    │
│  │                                          │                                    │
│  │  BUILT-IN GUARDRAILS:                    │                                    │
│  │    stop_loss, take_profit, max_hold,     │                                    │
│  │    reentry_cooldown, max_positions,      │                                    │
│  │    cash_check, max_entries_per_day       │                                    │
│  │                                          │                                    │
│  │  MODES:                                  │                                    │
│  │    --mode backtest  (historical test)    │                                    │
│  │    --mode signals   (live evaluation)    │                                    │
│  └──────────────────────────────────────────┘                                    │
│                                                                                  │
│  ┌──────────────────────────────────────────┐                                    │
│  │       rt_utils (real-time, intraday)     │                                    │
│  │  - live quote + options chain            │                                    │
│  │  - news sentiment (weighted avg)         │                                    │
│  │  - earnings + company overview           │                                    │
│  │  - LLM summary (ollama llama3.1:8b)     │                                    │
│  │  - email report                          │                                    │
│  └──────────────────────────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### Key Boundaries

- **Alpha Vantage API** — external paid API. Rate-limited (5 calls/min free, 75/min premium). Used by `pull_api_data`, `rt_utils`, `backfill_options`. Set `ALPHAVANTAGE_API_KEY` env var.
- **IBKR Gateway** — runs on a separate AWS EC2 instance. Access via SSH tunnel (`ssh -L 4001:127.0.0.1:4001`). Port 4001 = live, port 4002 = paper.
- **MySQL DB** — local database (Docker or native). All historical data lives here. Strategies read from exported CSVs, not the DB directly.
- **ollama** — local LLM inference server. Only needed for `rt_utils --summary`. Runs llama3.1:8b locally.
- **Google Drive** — overnight PDF upload via service account credentials. Only used by `cron.py`.

---

### 3. Nightly Cron Pipeline (`cron.py`)

The nightly cron job automates the data-refresh + screener + report pipeline.

```bash
python cron.py
```

What it does (in order):
1. Acquires a lock file (`/tmp/cron.lock`) to prevent concurrent runs
2. Pulls all API data: `python -m api_data.pull_api_data -w all`
3. Runs the screener: `python -m visualizations.screener --n_days 60 --data .`
4. Uploads the overnight PDF to Google Drive
5. Cleans up temp files (table images, screener CSVs, PDFs)

Requires: `service_account_credentials.json` in the project root for Google Drive upload.

---

### 4. Data Pipeline

#### 4.1 Pull All Data

```bash
python -m api_data.pull_api_data -w all
```
Pulls OHLCV, technical indicators, fundamentals, options, and economic data for all S&P500 + watchlist symbols into the DB.

Pull for specific symbols only (faster for testing):
```bash
python -m api_data.pull_api_data -s AAPL NVDA MSFT
```

#### 4.2 Rebuild Gold Table Only

```bash
python -m api_data.pull_api_data --gold-table-only
```
Rebuilds the joined feature table without re-fetching from the API. Useful after schema changes or manual data edits.

#### 4.3 Backfill Historical Options

```bash
python api_data/backfill_options.py -s AAPL NVDA --start_date 2024-01-01 --end_date 2025-01-01
```
Multi-threaded backfill of historical options data for specific symbols and date range.

#### 4.4 Data Validation

```bash
python api_data/validate_db.py
```
Check the MySQL database for duplicates, missing values, and integrity issues.

```bash
python api_data/validate.py -c all_data.csv
```
Validate a CSV data file for quality issues before training.

---

### 5. Screening & Visualization

#### 5.1 Run Screener

```bash
python -m visualizations.screener --n_days 60 --data .
```
Run the stock screener on all symbols, generate technical charts and an overnight PDF report.
- `--n_days` — lookback window for charts (default: 60)
- `--data` — directory containing data files

For specific symbols:
```bash
python -m visualizations.screener --symbols AAPL NVDA --n_days 30 --data .
```

---

### 6. Real-Time Utilities (`rt_utils`)

#### 6.1 Basic Quote + Options

```bash
python -m api_data.rt_utils AAPL
```
Fetches real-time quote and nearest monthly options chain.
- `--months-out 2` — expiration months out (default: 1)
- `--strikes 3` — strikes above/below ATM (default: 2)
- `--type call` — call, put, or both (default: both)

#### 6.2 News Sentiment

```bash
python -m api_data.rt_utils AAPL --news
```
Adds weighted news sentiment analysis across top 20 articles.

#### 6.3 Earnings + LLM Summary

```bash
python -m api_data.rt_utils AAPL --earnings-summary
```
Fetches earnings and company overview, generates local LLM summary of fundamentals.

#### 6.4 Full Report

```bash
python -m api_data.rt_utils AAPL --news --summary
```
Full report: quote, options, news sentiment, earnings overview, and combined LLM summary.

```bash
python -m api_data.rt_utils AAPL --news --summary --no-earnings
```
News-only LLM summary (skip earnings API calls — useful for daily runs after initial review).

#### 6.5 Email Report

```bash
python -m api_data.rt_utils AAPL --news --summary --email
```
Run full report and send formatted HTML email to configured recipient list.

---

### 7. ML Model Training (Stockformer)

#### 7.1 Smoke Test

```bash
python -m stockformer.smoke_test --data-path "./all_data_*.csv"
```
Quick 1-epoch smoke test to verify the training pipeline works end to end.

#### 7.2 Train All 9 Models

```bash
python -m stockformer.main --data-path "./all_data_*.csv"
```
Trains all 9 models: 3 horizons (3d, 10d, 30d) x 3 label modes (regression, binary, buckets).

Key arguments:
- `--horizon 3` — prediction horizon: 3, 10, or 30 days
- `--label-mode regression` — regression, binary, or buckets
- `--batch-size 64` — batch size (default: 64)
- `--epochs 15` — training epochs
- `--model-type encoder` — encoder (default) or cross_attention
- `--train-end-date 2025-10-31` — training data cutoff
- `--infer-start-date 2025-11-01` — inference data start

Train a single model:
```bash
python -m stockformer.main --data-path "./all_data_*.csv" \
    --horizon 3 --label-mode binary --epochs 50
```

#### 7.3 Parallel Training

```bash
./scripts/run_parallel_training.sh
```
Launches 3 parallel jobs (regression, binary, buckets), each handling 3 horizons sequentially. Reduces ~36h to ~12h.

```bash
./scripts/run_parallel_training.sh --cross-attention
```
Same but using market encoder + stock decoder with cross-attention architecture.
- `--encoder` — original bidirectional encoder (default)
- `--cross-attention` — cross-attention architecture

Fixed defaults in the script: TRAIN_END=2025-10-31, INFER_START=2025-11-01, BATCH_SIZE=64, EPOCHS=15.

Monitor progress:
```bash
tail -f logs/train_*_encoder_*.log
ps aux | grep stockformer
```

#### 7.4 Inference Only

```bash
python -m stockformer.main --data-path "./all_data_*.csv" \
    --horizon 3 --label-mode regression --infer-only --model-out model.pt
```
Run inference only on a previously trained model (skip training).

#### 7.5 Merge Predictions

```bash
python -m stockformer.merge_predictions --input-dir stockformer/output --output merged_predictions.csv
```
Merges all 9 prediction CSVs into a single file with all prediction columns.
- `--input-dir` — directory with prediction CSVs (default: output/)
- `--output` — output path (default: merged_predictions.csv)

---

### 8. Portfolio Filtering & Ranking

The portfolio ranker narrows the symbol universe before backtesting or live trading. Filters are composable — combine any of them.

#### 8.1 Rank by Sharpe

```bash
python strategies/portfolio.py --data all_data_0.csv --top-k-sharpe 20 --summary
```
- `--top-k-sharpe 20` — keep top 20 by Sharpe ratio
- `--summary` — print ranked table
- `--risk-free-rate 0.05` — risk-free rate for Sharpe (default: 0.0)

#### 8.2 Price Band Filter

```bash
python strategies/portfolio.py --data all_data_0.csv --price-above 50 --price-below 500 --summary
```

#### 8.3 Field Filter

```bash
python strategies/portfolio.py --data all_data_0.csv --filter-field beta --filter-below 1.5 --summary
```
Available fields: beta, rsi_14, pe_ratio, market_cap, or any numeric column in the data CSV.

```bash
python strategies/portfolio.py --data all_data_0.csv --filter-field rsi_14 --filter-above 30 --filter-below 70 --summary
```

#### 8.4 Watchlist

```bash
python strategies/portfolio.py --data all_data_0.csv --watchlist api_data/watchlist.csv --watchlist-mode filter --summary
```
- `--watchlist-mode sort` — watchlist symbols first, then rest (default)
- `--watchlist-mode filter` — watchlist symbols only

#### 8.5 Combined Pipeline + Save

```bash
python strategies/portfolio.py --data all_data_0.csv \
    --price-above 25 --filter-field beta --filter-below 1.5 --top-k-sharpe 15 \
    --output portfolio_picks.csv --summary
```
Output CSV has a `symbol` column and can be piped into backtests with `--symbols-file`.

---

### 9. Backtesting

#### 9.1 ML Prediction Strategy

```bash
python strategies/run_backtest.py \
    --predictions merged_predictions.csv \
    --symbols AAPL GOOGL MSFT \
    --start-date 2024-07-01 --end-date 2024-12-31
```
- Entry: `pred_class = 1` (positive return predicted)
- Exit: `pred_class = 0` (negative return predicted)

Key arguments:
- `--predictions` — path to stockformer predictions CSV (required)
- `--symbols AAPL NVDA` — explicit symbol list
- `--symbols-file portfolio_picks.csv` — load symbols from CSV (output of portfolio)
- `--use-all-symbols` — use all symbols in predictions file
- `--start-cash 100000` — initial cash (default: 100000)
- `--position-size 0.1` — fraction per position (default: 0.1 = 10%)
- `--output-trades trades.csv` — write trade log
- `--output-symbols symbols.csv` — write symbol list
- `--output-signals signals/pending_orders.csv` — write signal CSV for live execution

#### 9.2 Regression Momentum Strategy

```bash
python strategies/run_regression_momentum.py \
    --predictions merged_predictions.csv \
    --use-all-symbols \
    --start-date 2024-07-01 --end-date 2024-12-31
```
- Entry: `pred_reg_3d > 1%` AND `pred_reg_10d > 2%` AND `adx_signal > 0`
- Exit: `pred_reg_3d < 0` OR `cci_signal < 0` OR hold >= 13 days
- Min hold: 2 days

Same arguments as run_backtest.py plus the same portfolio filtering flags.

#### 9.3 Backtesting with Portfolio Filters

```bash
python strategies/run_regression_momentum.py \
    --predictions merged_predictions.csv --use-all-symbols \
    --portfolio-data all_data_0.csv \
    --price-above 25 --top-k-sharpe 15 \
    --start-date 2024-07-01 --end-date 2024-12-31 \
    --output-trades trades.csv --output-symbols filtered_symbols.csv
```

Using a pre-saved symbol list:
```bash
python strategies/run_regression_momentum.py \
    --predictions merged_predictions.csv \
    --symbols-file portfolio_picks.csv \
    --start-date 2024-07-01 --end-date 2024-12-31
```

#### 9.4 Bollinger Band Strategy

```bash
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' \
    --start-date 2024-01-01 --end-date 2024-06-30 \
    --watchlist api_data/watchlist.csv --watchlist-mode filter
```
- Entry: Price crosses below lower BB AND RSI <= 40
- Exit: Price reaches middle BB OR RSI > 70 OR hold >= 20 days
- Open positions are force-closed at backtest end

Key arguments:
- `--data-path` — path to price/indicator CSV(s) (supports globs)
- `--watchlist` — watchlist CSV for symbol filtering
- `--watchlist-mode filter` — only trade watchlist symbols (or `sort` for ordering)
- `--start-cash 100000` — initial cash (default: 100000)
- `--position-size 0.05` — fraction per position (default: 0.05 = 5%)
- `--max-hold-days 20` — max days to hold (default: 20)
- `--no-notify` — skip email notification

Daily watch mode (scan for crossovers, no backtest):
```bash
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' \
    --mode daily --lookback-days 2 \
    --watchlist api_data/watchlist.csv --watchlist-mode filter
```
- Lists which symbols crossed their bands in the last N trading days
- No account, no position tracking — just a scan and email report
- Email subject: `Bollinger Band (Daily Watch) | ...`

Email subjects by mode:
- Backtest: `Bollinger Band (Backtest) | 91 trades, 44.0% win rate, Sharpe -0.12`
- Daily: `Bollinger Band (Daily Watch) | 3 signal(s): 2 BUY, 1 SELL`

#### 9.6 Template Strategy (Build Your Own)

Copy `template_strategy.py` and fill in your entry/exit logic:
```bash
cp strategies/template_strategy.py strategies/my_strategy.py
# Edit my_strategy.py: rename class, fill in _check_entry_conditions() and _check_exit_custom()
# Update run_template.py import to point at your class
```

Backtest:
```bash
python strategies/run_template.py \
    --predictions data.csv --symbols AAPL NVDA \
    --start-date 2024-07-01 --end-date 2024-12-31
```

Built-in guardrails (class constants to tune):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `STOP_LOSS_PCT` | -5.0 | Exit at -5% loss (None to disable) |
| `TAKE_PROFIT_PCT` | 10.0 | Exit at +10% gain (None to disable) |
| `MAX_HOLD_DAYS` | 20 | Force exit after 20 days (None to disable) |
| `MIN_HOLD_DAYS` | 1 | Don't exit before 1 day |
| `ALLOW_REENTRY` | True | Allow re-entering after exit |
| `REENTRY_COOLDOWN_DAYS` | 3 | Days before same symbol can re-enter |
| `MAX_POSITIONS` | 10 | Max concurrent open positions |
| `MAX_ENTRIES_PER_DAY` | 5 | Max new entries per evaluation |

#### 9.7 Backtest Output

Every backtest runner prints:
- Per-symbol P&L breakdown (sorted by total P&L)
- Win/loss count and win rate per symbol
- Average hold days and average return per trade
- Sharpe ratio (annualized from trade-level returns)
- Portfolio-level summary (final value, total return)
- Open positions at backtest end

Output files:
- `--output-trades trades.csv` — full trade log (symbol, entry/exit dates, prices, P&L, hold days)
- `--output-symbols symbols.csv` — filtered symbol list (pipeable into other runners)
- `--output-signals signals/pending_orders.csv` — signal CSV for live execution bridge

---

### 10. Strategy → Execution Bridge (Signal Files)

The signal CSV is the contract between strategies and the IBKR executor. Strategies write pending signals; the executor reads, executes, and archives them.

#### 10.1 Signal CSV Format

Columns: `action, symbol, shares, price, strategy, reason, timestamp`
- `shares=0` → executor auto-sizes from account
- `price=0` → use live market price

#### 10.2 Writing Signals from Strategy

Backtest runners write signals with `--output-signals`:
```bash
python strategies/run_regression_momentum.py \
    --predictions merged_predictions.csv --symbols AAPL NVDA \
    --start-date 2024-07-01 --end-date 2024-12-31 \
    --output-signals signals/pending_orders.csv
```

Live signal generation (evaluate once, write CSV or nothing):
```bash
python strategies/run_template.py \
    --mode signals --predictions data.csv --symbols AAPL NVDA \
    --output-signals signals/pending_orders.csv
```

#### 10.3 Bollinger Shadow Strategy

Scans for Bollinger band crossovers and sends email notifications (shadow mode — doesn't execute by default).

**CSV mode** (scan all_data files):
```bash
python strategies/bollinger_shadow_strategy.py --data-path all_data.csv
```
- `--skip-live` — test without Gateway connection
- `--no-notify` — skip email notifications
- `--summary-only` — one summary email instead of per-signal emails
- `--execute AAPL` — actually execute for listed symbols (comma-separated)
- `--lookback-days 2` — days to look back for signals (default: 2)

**Realtime mode** (fetch live data from Alpha Vantage API):
```bash
python strategies/bollinger_shadow_strategy.py \
    --mode realtime \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --skip-live --summary-only
```
- Fetches live quote + BBANDS + RSI from Alpha Vantage for each watchlist symbol (3 API calls per symbol)
- Runs the same crossover logic against live data
- Writes signal CSV to `signals/pending_orders.csv` if anything triggers
- Email subject: `Bollinger Band (Real Time) | 5 signal(s)`
- `--watchlist` is **required** for realtime mode

With portfolio post-filter (CSV mode):
```bash
python strategies/bollinger_shadow_strategy.py --data-path all_data.csv --skip-live --no-notify \
    --portfolio-data all_data_0.csv --price-above 50 --top-k-sharpe 10
```

---

### 11. Live Execution (IBKR)

#### 11.1 Trading Mode Safety Gate

Before any live trading, you must manually edit `ibkr/trading_mode.conf`:
```
TRADING_MODE=paper    ← default (safe)
TRADING_MODE=live     ← enables real money trading
```
The executor refuses to connect to live ports unless this file explicitly says `live`. Change it back to `paper` after your session.

#### 11.2 Test Gateway Connectivity

```bash
python ibkr/test_gateway.py --balance --portfolio
```
- `--host 127.0.0.1` — Gateway host (default: 127.0.0.1)
- `--port 4001` — Gateway port (default: 4001)
- `--all` — test balance + portfolio
- `--buy` / `--sell` — test order (paper only)
- `--symbol NKE` — symbol for test orders (default: NKE)
- `--shares 1` — shares for test orders (default: 1)
- `--dry-run` — simulate without executing
- `--test-notify` — test email/SMS notifications

#### 11.3 Execute Signal Files

Dry run (preview without executing):
```bash
python -m ibkr.execute_signals --signals signals/pending_orders.csv --dry-run
```

Paper trading (one-shot):
```bash
python -m ibkr.execute_signals --signals signals/pending_orders.csv
```
- `--client-id 10` — IBKR client ID (default: 10)
- `--position-size 0.10` — fraction per position (default: 0.10)
- `--max-positions 10` — max concurrent positions (default: 10)
- `--max-daily-loss 5000` — max daily loss in dollars (default: 5000)
- `--gateway` — use IB Gateway instead of TWS

Live trading:
```bash
python -m ibkr.execute_signals --signals signals/pending_orders.csv --live
```
Requires `TRADING_MODE=live` in `ibkr/trading_mode.conf`.

Watch mode (poll for new signal files):
```bash
python -m ibkr.execute_signals --watch --signals-dir signals/ --interval 30
```
- `--interval 30` — seconds between polls (default: 30)
- `--archive-dir signals/executed/` — where to move processed files

#### 11.4 Live Strategy Runner

Continuous evaluation (every 5 min):
```bash
python ibkr/run_live_strategy.py --symbols AAPL NVDA MSFT
```

Run once and exit (for cron):
```bash
python ibkr/run_live_strategy.py --symbols AAPL NVDA MSFT --once
```

Live mode:
```bash
python ibkr/run_live_strategy.py --symbols AAPL NVDA MSFT --live --once
```
- `--interval 5` — minutes between evaluations (default: 5)
- `--predictions-dir` — directory with prediction files
- `--gateway` — use IB Gateway instead of TWS
- `--position-size 0.10` — fraction per position (default: 0.10)
- `--max-positions 10` — max concurrent positions (default: 10)
- `--max-daily-loss 5000` — max daily loss (default: 5000)

---

### 12. Reinforcement Learning (Experimental)

```bash
python bargyms/train.py
```
Train a PPO reinforcement learning agent on the custom trading environment.

---

### 13. Typical Workflows

#### Daily Morning

```bash
python cron.py                                              # refresh data + overnight PDF
python -m api_data.rt_utils AAPL --news --summary --email   # real-time report for key symbols
```

#### Train → Infer → Backtest

```bash
# 1. Train all models (~12h parallel)
./scripts/run_parallel_training.sh --cross-attention

# 2. Merge predictions
python -m stockformer.merge_predictions --input-dir stockformer/output --output merged_predictions.csv

# 3. Filter portfolio
python strategies/portfolio.py --data all_data_0.csv \
    --price-above 25 --top-k-sharpe 15 --output portfolio_picks.csv --summary

# 4. Backtest with filtered symbols
python strategies/run_regression_momentum.py \
    --predictions merged_predictions.csv --symbols-file portfolio_picks.csv \
    --start-date 2024-07-01 --end-date 2024-12-31 \
    --output-trades trades.csv --output-signals signals/pending_orders.csv
```

#### Backtest → Paper Trade

```bash
# 1. Run backtest, output signals for best-performing symbols
python strategies/run_regression_momentum.py \
    --predictions merged_predictions.csv --symbols-file portfolio_picks.csv \
    --start-date 2024-07-01 --end-date 2024-12-31 \
    --output-signals signals/pending_orders.csv

# 2. Dry run to preview what would execute
python -m ibkr.execute_signals --signals signals/pending_orders.csv --dry-run

# 3. Execute on paper account
python -m ibkr.execute_signals --signals signals/pending_orders.csv

# 4. Or use watch mode for continuous signal pickup
python -m ibkr.execute_signals --watch --signals-dir signals/ --interval 30
```

#### Live Signal Loop (Hourly)

```bash
# Strategy evaluates once, writes signal CSV if there's an action, writes nothing if hold
python strategies/run_template.py \
    --mode signals --predictions data.csv --symbols-file portfolio_picks.csv \
    --output-signals signals/pending_orders.csv

# Executor picks up and places orders
python -m ibkr.execute_signals --signals signals/pending_orders.csv
```

#### Bollinger Band Pipeline (Backtest → Scan → Paper Trade)

```bash
# 1. Full backtest — per-symbol P&L, win rate, Sharpe + email summary
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' \
    --start-date 2024-01-01 --end-date 2024-06-30 \
    --watchlist api_data/watchlist.csv --watchlist-mode filter

# 2. Daily watch — scan for crossovers in last 2 days + email
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' \
    --mode daily --lookback-days 2 \
    --watchlist api_data/watchlist.csv --watchlist-mode filter

# 3. Real-time scan — live Alpha Vantage data + email + signal CSV
python strategies/bollinger_shadow_strategy.py \
    --mode realtime \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --skip-live --summary-only

# 4. Paper trade execution (requires SSH tunnel + Gateway)
python -m ibkr.execute_signals --signals signals/pending_orders.csv --gateway --client-id 10
```

#### Real-Time Scan + Execute Loops

For continuous intraday trading, run the scanner and executor as two parallel loops.
The scanner polls Alpha Vantage on an interval and writes signals; the executor watches
for the signal file and sends orders to IBKR the moment it appears.

**Terminal 1 — Scanner** (scans every 15 min during market hours):
```bash
./scripts/rt_scan_loop.sh --interval 15
```

**Terminal 2 — Executor** (polls every 2s for signals, paper trading):
```bash
./scripts/rt_execute_loop.sh
```

Both scripts enforce market hours (9:30am–4:00pm ET, Mon–Fri) and sleep outside trading hours.

Scanner options:
- `--interval N` — minutes between scans (default: 15, can go to 1 or less later)
- `--watchlist FILE` — override watchlist (default: `api_data/watchlist.csv`)
- `--dry-run` — print what would run without executing
- `--once` — single scan, no loop

Executor options:
- `--poll N` — seconds between file checks (default: 2)
- `--dry-run` — log what would execute, archive signal file, don't trade
- `--live` — live trading (default: paper via `--gateway`)
- `--client-id N` — IBKR client ID (default: 10)
- `--once` — check once and exit

Logs: `logs/rt_scan.log` and `logs/rt_execute.log`
Executed signals archived to: `signals/executed/orders_YYYYMMDD_HHMMSS.csv`

#### Going Live

```bash
# 1. Edit the safety gate
# Change TRADING_MODE=paper to TRADING_MODE=live in ibkr/trading_mode.conf

# 2. Execute with --live flag
python -m ibkr.execute_signals --signals signals/pending_orders.csv --live

# 3. After session, change trading_mode.conf back to paper
```

---

### Appendix A: Shell Script Helpers

All scripts live in `scripts/` and are ready to run. They wrap common multi-step workflows into single commands.

#### A.1 `scripts/infer_merge.sh` — Inference + Merge

Run inference on all 9 pre-trained models in parallel, then merge predictions into one CSV.

```bash
./scripts/infer_merge.sh                    # encoder models (default)
./scripts/infer_merge.sh --cross-attention  # cross-attention models
```

What it does:
1. Launches 3 parallel inference jobs (regression, binary, buckets) — each runs 3 horizons
2. Waits for all jobs to finish
3. Merges all prediction CSVs into `merged_predictions.csv`

Defaults: DATA_PATH=`./all_data_*.csv`, INFER_START=2025-11-01, BATCH_SIZE=64.

#### A.2 `scripts/train_infer_merge.sh` — Full Training Pipeline

Train all 9 models from scratch in parallel, then merge predictions. The "weekly retrain" script.

```bash
./scripts/train_infer_merge.sh                    # encoder (default)
./scripts/train_infer_merge.sh --cross-attention  # cross-attention
```

What it does:
1. Launches 3 parallel training jobs (regression, binary, buckets) — each trains 3 horizons
2. Waits for all jobs to finish (~12h)
3. Merges all prediction CSVs into `merged_predictions.csv`

Defaults: TRAIN_END=2025-10-31, INFER_START=2025-11-01, BATCH_SIZE=64, EPOCHS=15.

Monitor: `tail -f logs/train_*_encoder_*.log`

#### A.3 `scripts/backtest_pipeline.sh` — Portfolio Filter + Backtest

End-to-end backtest: filter portfolio, run strategy, output trades and signals.

```bash
# Defaults (regression_momentum, all_data_0.csv, top 15 sharpe)
./scripts/backtest_pipeline.sh

# ML prediction strategy with custom dates
./scripts/backtest_pipeline.sh --strategy ml_prediction \
    --start-date 2024-01-01 --end-date 2024-12-31

# Override predictions file and portfolio data
./scripts/backtest_pipeline.sh --predictions my_preds.csv \
    --portfolio-data all_data_1.csv --top-k 20 --price-above 50
```

Arguments:
- `--strategy regression_momentum` or `ml_prediction` (default: regression_momentum)
- `--predictions FILE` — predictions CSV (default: merged_predictions.csv)
- `--portfolio-data FILE` — portfolio data CSV (default: all_data_0.csv)
- `--top-k N` — top K by Sharpe (default: 15)
- `--price-above N` — minimum price filter (default: 25)
- `--start-date`, `--end-date` — backtest date range
- `--start-cash N` — initial cash (default: 100000)

What it does:
1. Runs `portfolio.py` to filter and save `portfolio_picks.csv`
2. Runs the selected strategy backtest with the filtered symbols
3. Outputs `trades.csv` and `signals/pending_orders.csv`

#### A.4 `scripts/daily_signals.sh` — Signal Generation (for cron)

Generate fresh signals from the latest predictions. Designed for cron or manual hourly runs.

```bash
./scripts/daily_signals.sh                  # generate signals only
./scripts/daily_signals.sh --execute        # generate + execute (paper)
./scripts/daily_signals.sh --execute --live # generate + execute (live)
```

Arguments:
- `--execute` — also run `ibkr.execute_signals` after generating
- `--live` — use live trading mode (requires trading_mode.conf=live)
- `--strategy regression_momentum` or `ml_prediction` (default: regression_momentum)
- `--predictions FILE` — predictions CSV (default: merged_predictions.csv)
- `--symbols-file FILE` — symbol list (default: portfolio_picks.csv)

Cron example (run every hour during market hours):
```bash
30 9-15 * * 1-5 ~/proj/bar_fly_trading/scripts/daily_signals.sh >> ~/proj/bar_fly_trading/logs/daily_signals.log 2>&1
```

#### A.5 `scripts/rt_report.sh` — Multi-Symbol Real-Time Report

Loop over portfolio symbols and run `rt_utils` for each. Morning review script.

```bash
./scripts/rt_report.sh                            # uses portfolio_picks.csv
./scripts/rt_report.sh --symbols AAPL NVDA MSFT   # explicit symbols
./scripts/rt_report.sh --email                     # send email for each
./scripts/rt_report.sh --news-only                 # skip earnings (faster)
```

Arguments:
- `--symbols SYM1 SYM2 ...` — explicit symbol list
- `--symbols-file FILE` — CSV with symbols (default: portfolio_picks.csv)
- `--email` — send email report for each symbol
- `--news-only` — skip earnings API calls (use `--no-earnings`)
- `--months-out N` — options expiration months out
- `--strikes N` — strikes above/below ATM

Rate limiting: 15s pause between symbols (Alpha Vantage free tier: 5 calls/min).
