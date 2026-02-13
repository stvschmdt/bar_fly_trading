## bar_fly_trading - End to End Tutorial

### 1. Setup

#### 1.1 Infrastructure

The system runs across two machines connected via Tailscale VPN:

| Machine | Role | Key Services |
|---------|------|-------------|
| **EC2** (AWS) | Data, execution, web | MySQL, IBKR Gateway, nginx, Alpha Vantage pulls |
| **DGX** (GPU) | ML training, inference, LLM | PyTorch (CUDA), ollama, stockformer models |

Tailscale IPs (stable, won't change):
```
EC2:  100.96.238.58   (sschmidt@100.96.238.58)
DGX:  100.x.x.x       (stvschmdt@dgx)
```

SCP between machines:
```bash
# DGX -> EC2
scp ~/proj/bar_fly_trading/merged_predictions.csv sschmidt@100.96.238.58:~/bar_fly_trading/

# EC2 -> DGX
scp sschmidt@100.96.238.58:~/bar_fly_trading/all_data_*.csv ~/proj/bar_fly_trading/
```

#### 1.2 Database (EC2)

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
sudo systemctl enable mysqld   # auto-start on reboot
sudo mysql -u root -e "CREATE DATABASE bar_fly_trading; ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'my-secret-pw'; FLUSH PRIVILEGES;"
```

Option C: Clone an existing database
```bash
# On source server
mysqldump -u root -p bar_fly_trading > bar_fly_trading_dump.sql

# Copy and import
scp user@source-server:bar_fly_trading_dump.sql .
mysql -u root -pmy-secret-pw bar_fly_trading < bar_fly_trading_dump.sql
```

#### 1.3 Environment Variables

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

#### 1.4 Dependencies

```bash
pip install -r requirements.txt
```

For LLM summaries (DGX only):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
pip install ollama
```

---

### 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES                               │
│                                                                         │
│  ┌───────────────────┐        ┌───────────────────────────────────┐     │
│  │  Alpha Vantage API │        │      IBKR Gateway (EC2)          │     │
│  │  (paid, premium)   │        │      Port 4001 (live)            │     │
│  │                    │        │      Port 4002 (paper)           │     │
│  │  - OHLCV prices    │        │                                   │     │
│  │  - Technicals      │        │  - Account balances              │     │
│  │  - Fundamentals    │        │  - Portfolio positions            │     │
│  │  - Options chains  │        │  - Order placement/execution     │     │
│  │  - Economic data   │        │  - Real-time market data         │     │
│  │  - News sentiment  │        │                                   │     │
│  │  - Earnings        │        │                                   │     │
│  └────────┬──────────┘        └──────────────┬────────────────────┘     │
│           │                                   │                          │
└───────────┼───────────────────────────────────┼──────────────────────────┘
            │ REST API                          │ IB API (TCP)
            │                                   │
┌───────────┼───────────────────────────────────┼──────────────────────────┐
│           │              EC2                   │                          │
│           ▼                                   │                          │
│  ┌──────────────────┐    ┌──────────────┐     │                          │
│  │  ec2_nightly.sh   │───>│   MySQL DB   │     │                          │
│  │  Step 1: data pull│    │              │     │                          │
│  │  Step 2: screener │    │ core_stock   │     │                          │
│  │  Step 3: web data │    │ gold_table   │     │                          │
│  └──────────────────┘    └──────┬───────┘     │                          │
│                                  │             │                          │
│                                  ▼             │                          │
│  ┌──────────────────┐    ┌──────────────┐     │                          │
│  │  rt_scan_loop.sh  │───>│all_data_*.csv│     │                          │
│  │  (market hours)   │    └──────────────┘     │                          │
│  │  4 strategies     │           │             │                          │
│  │  every 15 min     │           │ SCP         │                          │
│  └────────┬─────────┘           │             │                          │
│           │                      │             │                          │
│           ▼                      │             │                          │
│  ┌──────────────────┐           │             │                          │
│  │ pending_orders.csv│           │             │                          │
│  └────────┬─────────┘           │             │                          │
│           │                      │             │                          │
│           ▼                      │             │                          │
│  ┌──────────────────┐           │             │                          │
│  │ execute_signals   │───────────┼─────────────┘                          │
│  │ (IBKR bridge)     │           │                                        │
│  └──────────────────┘           │                                        │
└──────────────────────────────────┼────────────────────────────────────────┘
                                   │
              ─────── Tailscale VPN ───────
                                   │
┌──────────────────────────────────┼────────────────────────────────────────┐
│                                  │              DGX (GPU)                  │
│                                  ▼                                        │
│  ┌──────────────────┐    ┌──────────────────┐                            │
│  │  infer_merge.sh   │───>│   stockformer/    │                            │
│  │  9 models in      │    │   9 models:       │                            │
│  │  parallel (~2min) │    │   3 horizon x     │                            │
│  │                   │    │   3 label mode     │                            │
│  │  Auto-detects     │    │                    │                            │
│  │  architecture     │    │  model_reg_3d.pt   │                            │
│  │  from checkpoint  │    │  model_bin_3d.pt   │                            │
│  └──────────────────┘    │  model_buck_3d.pt  │                            │
│           │               │  ... (9 total)     │                            │
│           ▼               └──────────────────┘                            │
│  ┌──────────────────┐                                                     │
│  │merged_predictions │──── SCP ────> EC2                                  │
│  │.csv               │                                                     │
│  └──────────────────┘                                                     │
│                                                                           │
│  ┌──────────────────┐                                                     │
│  │  dgx_nightly.sh   │    ┌──────────────────┐                            │
│  │  1. Pull JSONs    │───>│   ollama          │                            │
│  │  2. LLM summaries │    │   llama3.1:8b     │                            │
│  │  3. Push JSONs    │    │   AI stock reports │                            │
│  └──────────────────┘    └──────────────────┘                            │
└───────────────────────────────────────────────────────────────────────────┘
```

#### Key Boundaries

- **Alpha Vantage API** — paid API. Rate-limited (5 calls/min free, 75/min premium). Set `ALPHAVANTAGE_API_KEY` env var.
- **IBKR Gateway** — runs on EC2. Port 4001 = live, port 4002 = paper. Only uses IB Gateway (never TWS).
- **MySQL DB** — runs on EC2. All historical data lives here. Strategies read from exported CSVs, not the DB directly.
- **ollama** — runs on DGX. LLM inference for AI stock summaries. Uses llama3.1:8b (GPU) or llama3.2:1b (CPU fallback).
- **Tailscale** — mesh VPN connecting EC2 and DGX with stable IPs. SCP/SSH between machines.

---

### 3. Nightly Pipeline (Happy Path)

This is the daily automated pipeline that runs after market close. Each step produces files that the next step depends on.

```
6:00pm ET          7:00pm ET          7:15pm ET          7:30pm ET
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
┌────────┐      ┌────────────┐    ┌──────────┐      ┌────────────┐
│ EC2    │      │ DGX        │    │ DGX→EC2  │      │ DGX        │
│ Data   │─────>│ Inference  │───>│ SCP      │─────>│ Website    │
│ Pull   │      │ + Merge    │    │ Preds    │      │ LLM Reports│
└────────┘      └────────────┘    └──────────┘      └────────────┘
 Step 1           Step 2           Step 3             Step 4
```

---

#### 3.1 Step 1: Data Pull (EC2)

**Script:** `scripts/ec2_nightly.sh`
**Where:** EC2
**Cron:** `0 18 * * 1-5` (6pm ET weekdays)

```bash
# Full pipeline (data + screener PDF + website data)
bash scripts/ec2_nightly.sh

# Just data pull
bash scripts/ec2_nightly.sh --step data

# Just screener PDF + Drive upload
bash scripts/ec2_nightly.sh --step pdf

# Just website data refresh (sector map, symbols, history)
bash scripts/ec2_nightly.sh --step web
```

**What it does:**
1. **Data pull** — `python -m api_data.pull_api_data -w all` — pulls OHLCV, technicals, fundamentals, options, economic data for all S&P500 + watchlist (~550 symbols) into MySQL, then exports gold table to `all_data_*.csv` (37 files, ~15 symbols each)
2. **Screener PDF** — generates overnight technical charts + uploads to Google Drive
3. **Website data** — builds sector map, populates symbol JSONs, generates chart history for the webapp

**Produces:**
| File | Description |
|------|------------|
| `all_data_0.csv` ... `all_data_36.csv` | Gold table CSVs (37 files, ~97 columns each) |
| `overnight_*.pdf` | Screener PDF (uploaded to Google Drive) |
| `/var/www/bft/data/*.json` | Website symbol data |

**Cron setup:**
```bash
# /etc/crontab or crontab -e on EC2
0 18 * * 1-5 cd /home/sschmidt/bar_fly_trading && bash scripts/ec2_nightly.sh >> /var/log/bft/nightly.log 2>&1
```

**Dependencies to start:** MySQL running, `ALPHAVANTAGE_API_KEY` set, `MYSQL_PASSWORD` set

##### Fixes / Backfill / Manual Runs

Pull specific symbols only (faster for fixing gaps):
```bash
python -m api_data.pull_api_data -s AAPL NVDA MSFT
```

Rebuild gold table without re-fetching from API:
```bash
python -m api_data.pull_api_data --gold-table-only
```

Check for stale symbols in MySQL and re-pull any that are behind:
```bash
python scripts/check_stale_symbols.py              # report only
python scripts/check_stale_symbols.py --fix         # re-pull stale symbols
python scripts/check_stale_symbols.py --fix --rebuild-csv  # re-pull + rebuild CSVs
```

Backfill historical options data:
```bash
python api_data/backfill_options.py -s AAPL NVDA --start_date 2024-01-01 --end_date 2025-01-01
```

Validate database integrity:
```bash
python api_data/validate_db.py          # check MySQL for dupes/missing values
python api_data/validate.py -c all_data.csv  # validate a CSV before training
```

If MySQL won't start after reboot:
```bash
sudo systemctl start mysqld
sudo systemctl enable mysqld
# If tmpdir errors: check /etc/my.cnf, ensure tmpdir exists with mysql:mysql ownership
```

---

**Dependencies for Step 2:** `all_data_*.csv` files on DGX (SCP from EC2 or already present from prior pull)

---

#### 3.2 Step 2: Inference & Merge (DGX)

**Script:** `scripts/infer_merge.sh`
**Where:** DGX
**Cron:** `0 19 * * 1-5` (7pm ET, 1 hour after EC2 nightly starts — or trigger after step 1 completes)

```bash
# Encoder models (default)
bash scripts/infer_merge.sh

# Cross-attention models
bash scripts/infer_merge.sh --cross-attention
```

**What it does:**
1. Launches 3 parallel inference jobs (regression, binary, buckets) — each runs 3 horizons (3d, 10d, 30d)
2. **Auto-detects model architecture from checkpoint** — d_model, num_layers, dim_feedforward, nhead, num_buckets are read from the saved weights. No manual `--d-model` needed.
3. Waits for all 9 jobs to finish (~2 minutes on GPU)
4. Merges all 9 prediction CSVs into `merged_predictions.csv`

**Architecture auto-detection example output:**
```
Loading model from: stockformer/output/models/model_bin_3d.pt
Auto-detected architecture from checkpoint: d_model: 128 -> 64, num_layers: 3 -> 2, dim_feedforward: 256 -> 128
```

**Produces:**
| File | Description |
|------|------------|
| `stockformer/output/predictions/pred_reg_3d.csv` | Regression 3-day predictions |
| `stockformer/output/predictions/pred_bin_10d.csv` | Binary 10-day predictions |
| ... (9 total) | All horizon/label combinations |
| `merged_predictions.csv` | Merged file with all prediction columns |

**Defaults:** `DATA_PATH=./all_data_*.csv`, `INFER_START=2025-07-01`, `BATCH_SIZE=64`

**Monitor:**
```bash
tail -f logs/infer_reg_encoder_*.log
tail -f logs/infer_bin_encoder_*.log
tail -f logs/infer_buck_encoder_*.log
```

##### Fixes / Single Model Runs

Run inference for a single model:
```bash
python -m stockformer.main --data-path './all_data_*.csv' \
    --horizon 3 --label-mode binary \
    --model-out stockformer/output/models/model_bin_3d.pt \
    --output-csv stockformer/output/predictions/pred_bin_3d.csv \
    --infer-only
```

Merge predictions manually (after re-running a subset):
```bash
python -m stockformer.merge_predictions \
    --input-dir stockformer/output/predictions \
    --output merged_predictions.csv
```

Check what architecture a model was trained with:
```bash
python -c "
import torch
sd = torch.load('stockformer/output/models/model_bin_3d.pt', map_location='cpu', weights_only=True)
print(f'd_model: {sd[\"input_proj.weight\"].shape[0]}')
print(f'num_layers: {max(int(k.split(\".\")[2]) for k in sd if k.startswith(\"encoder.layers.\")) + 1}')
print(f'dim_ff: {sd[\"encoder.layers.0.linear1.weight\"].shape[0]}')
"
```

---

**Dependencies for Step 3:** `merged_predictions.csv` on DGX (produced by step 2)

---

#### 3.3 Step 3: Transfer Predictions to EC2

**Where:** DGX -> EC2 (SCP)
**How:** Manual or scripted

```bash
scp ~/proj/bar_fly_trading/merged_predictions.csv sschmidt@100.96.238.58:~/bar_fly_trading/merged_predictions.csv
```

**Produces:** `merged_predictions.csv` on EC2 at `~/bar_fly_trading/`

**Verify on EC2:**
```bash
ls -lh ~/bar_fly_trading/merged_predictions.csv
head -1 ~/bar_fly_trading/merged_predictions.csv | tr ',' '\n' | head -20
```

##### Fixes

If SCP fails (DNS/network):
```bash
# Test Tailscale connectivity
ssh sschmidt@100.96.238.58 "echo OK"

# If DNS broken on EC2 (Tailscale MagicDNS):
# On EC2: echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

---

**Dependencies for Step 4:** Symbol JSONs on EC2 (produced by step 1, step 3 web), `all_data_*.csv` on DGX

---

#### 3.4 Step 4: Website LLM Reports (DGX)

**Script:** `scripts/dgx_nightly.sh`
**Where:** DGX
**Cron:** `0 19 * * 1-5` (7pm ET, after EC2 nightly)

```bash
# Full pipeline: pull JSONs from EC2, run LLM summaries, push back
bash scripts/dgx_nightly.sh

# Individual steps
bash scripts/dgx_nightly.sh --step pull      # pull JSONs from EC2
bash scripts/dgx_nightly.sh --step reports   # run generate_reports (ollama)
bash scripts/dgx_nightly.sh --step push      # push JSONs back to EC2
```

**What it does:**
1. **Pull** — SCPs symbol JSONs from EC2 (`/var/www/bft/data/*.json`) to DGX
2. **Reports** — Runs `generate_reports` with ollama (llama3.1:8b) to generate AI stock summaries for each symbol. Auto-detects GPU availability — falls back to llama3.2:1b if training is running.
3. **Push** — SCPs updated JSONs back to EC2

**Produces:** Updated `*.json` files on EC2 with AI-generated summaries

**Cron setup:**
```bash
# DGX crontab
0 19 * * 1-5 cd ~/proj/bar_fly_trading && bash scripts/dgx_nightly.sh >> logs/dgx_nightly.log 2>&1
```

##### Fixes / Manual Runs

Generate reports for specific symbols only:
```bash
BFT_DATA_DIR=webapp_data python -m webapp.backend.generate_reports \
    --csv-pattern "./all_data_*.csv" --symbols AAPL,NVDA,JPM
```

Change LLM model:
```bash
BFT_OLLAMA_MODEL=llama3.1:8b bash scripts/dgx_nightly.sh --step reports
```

If ollama is not running:
```bash
ollama serve &   # start in background
ollama list      # verify models are available
```

---

### 4. Real-Time Scan & Execution (Market Hours)

During market hours (9:30am–4:00pm ET, Mon–Fri), two loops run on EC2:

```
┌─────────────────┐         ┌─────────────────┐
│  Scanner         │         │  Executor        │
│  rt_scan_loop.sh │────────>│  rt_execute_loop │
│  every 15 min    │ signal  │  polls every 2s  │
│  4 strategies    │  CSV    │  sends to IBKR   │
└─────────────────┘         └─────────────────┘
```

**Dependencies:** `all_data_*.csv` (from step 1), `merged_predictions.csv` (from step 3), `ALPHAVANTAGE_API_KEY`

#### 4.1 Scanner

```bash
# Default: 15 min interval, all 4 strategies
./scripts/rt_scan_loop.sh

# Custom interval
./scripts/rt_scan_loop.sh --interval 5

# Specific strategies only
./scripts/rt_scan_loop.sh --strategies "bollinger oversold_bounce"

# Dry run (print what would run)
./scripts/rt_scan_loop.sh --dry-run

# Single scan, no loop
./scripts/rt_scan_loop.sh --once
```

**Active strategies (4):**
| Strategy | Entry Logic | Max Hold | Instrument |
|----------|-----------|----------|------------|
| bollinger | Price < lower BB AND RSI <= 40 | 20 days | stock |
| oversold_bounce | RSI < 30 bounce pattern | 5 days | option |
| oversold_reversal | RSI reversal + volume | 7 days | stock |
| low_bounce | 52-week low bounce | 30 days | stock |

Each scan cycle:
1. Fetches bulk realtime quotes from Alpha Vantage (1 call per 100 symbols)
2. Merges with overnight data (technicals + ML predictions)
3. Runs all 4 strategies against merged data
4. Aggregates signals into `signals/pending_orders.csv`

**Scanner options:**
- `--interval N` — minutes between scans (default: 15)
- `--strategies "list"` — space-separated strategy names
- `--watchlist FILE` — override watchlist (default: `api_data/watchlist.csv`)
- `--no-notify` — suppress email notifications
- `--dry-run` — print commands without running
- `--once` — single scan, no loop
- `--execute` — also execute signals immediately after scan
- `--buy-only` — skip SELL signals for symbols we don't hold

#### 4.2 Executor

```bash
# Default: poll every 2s, paper trading
./scripts/rt_execute_loop.sh

# Custom poll interval
./scripts/rt_execute_loop.sh --poll 1

# Dry run (preview only)
./scripts/rt_execute_loop.sh --dry-run

# Live trading (requires ibkr/trading_mode.conf=live)
./scripts/rt_execute_loop.sh --live

# Single check and exit
./scripts/rt_execute_loop.sh --once
```

**Executor options:**
- `--poll N` — seconds between file checks (default: 2)
- `--dry-run` — log what would execute, don't trade
- `--live` — live trading (default: paper via `--gateway`)
- `--client-id N` — IBKR client ID (default: 10)
- `--once` — check once and exit

**Logs:** `logs/rt_scan.log`, `logs/rt_execute.log`
**Archives:** `signals/executed/orders_YYYYMMDD_HHMMSS.csv`

#### 4.3 Running as Background Services

```bash
# On EC2 — scanner
nohup bash scripts/rt_scan_loop.sh >> logs/rt_scan.log 2>&1 &

# On EC2 — executor
nohup bash scripts/rt_execute_loop.sh >> logs/rt_execute.log 2>&1 &
```

##### Fixes / Manual Strategy Runs

Run a single strategy manually:
```bash
# Bollinger — backtest
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --start-date 2025-08-01 --end-date 2026-01-31

# Bollinger — live scan (no execution)
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --mode live --skip-live --lookback-days 1 --summary-only --no-notify \
    --output-signals signals/pending_orders.csv

# Oversold bounce — live scan with options
python strategies/run_oversold_bounce.py \
    --data-path 'all_data_*.csv' \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --mode live --skip-live --lookback-days 1 --summary-only --no-notify \
    --instrument-type option --output-signals signals/pending_orders.csv
```

Stock vs options mode — each strategy can route signals independently:
```bash
# Override instrument type for any strategy
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' --mode live --skip-live \
    --instrument-type option --output-signals signals/pending_orders.csv
```

---

### 5. Email & Sentiment Reports

The full-featured real-time analysis tool. Fetches live quotes, options chains, news sentiment, earnings data, and generates LLM-powered summaries — all packaged into HTML emails.

#### 5.1 Quick Quote + Options

```bash
python -m api_data.rt_utils AAPL
```
Fetches real-time quote and nearest monthly options chain.

#### 5.2 News Sentiment

```bash
python -m api_data.rt_utils AAPL --news
```
Adds weighted news sentiment analysis across top 20 articles.

#### 5.3 Earnings + Fundamentals Summary

```bash
python -m api_data.rt_utils AAPL --earnings-summary
```
Fetches earnings and company overview, generates LLM summary of fundamentals (requires ollama).

#### 5.4 Sector Analysis

```bash
python -m api_data.rt_utils AAPL --sector-analysis
```
Analyzes sector ETF performance and generates LLM sector synopsis.

#### 5.5 The Full Report (Everything)

```bash
python -m api_data.rt_utils AAPL \
    --news \
    --summary \
    --data-path 'all_data_*.csv' \
    --months-out 2 \
    --strikes 5 \
    --type both \
    --email
```

This runs the complete pipeline:
1. Real-time quote (price, change, volume)
2. Options chain (2 months out, 5 strikes above/below ATM, calls + puts)
3. News sentiment (weighted average across 20 articles)
4. Earnings + company overview
5. LLM summary combining all of the above (ollama llama3.1:8b)
6. Technical indicators from CSV data
7. Sends formatted HTML email to all configured recipients

#### 5.6 Multi-Symbol Batch (The Morning Briefing)

```bash
# Report on your whole portfolio, emailed
./scripts/rt_report.sh --email

# Explicit symbols
./scripts/rt_report.sh --symbols AAPL NVDA MSFT GOOGL AMZN --email

# News-only (skip earnings API calls — faster)
./scripts/rt_report.sh --symbols AAPL NVDA --email --news-only

# Custom options view
./scripts/rt_report.sh --symbols AAPL --email --months-out 3 --strikes 8
```

Rate limiting: 15s pause between symbols (Alpha Vantage).

#### 5.7 News-Only Mode (Skip Earnings)

```bash
python -m api_data.rt_utils AAPL --news --summary --no-earnings --email
```
Faster daily check — skips the earnings/fundamentals API calls and does a news-only LLM summary.

#### 5.8 Multi-Symbol in One Call

```bash
python -m api_data.rt_utils AAPL,NVDA,MSFT --news --summary --email
```
Comma-separated symbols — runs a report for each and sends individual emails.

#### 5.9 Email to Specific Recipient

```bash
python -m api_data.rt_utils AAPL --news --summary --email --email-to someone@example.com
```
Overrides `IBKR_NOTIFY_EMAIL` and sends only to the specified address.

#### All rt_utils Flags

| Flag | Default | Description |
|------|---------|------------|
| `symbol` (positional) | required | Ticker(s) — single or comma-separated |
| `--months-out N` | 1 | Options expiration months out |
| `--strikes N` | 2 | Strikes above/below ATM |
| `--type` | both | call, put, or both |
| `--news` | off | Fetch news sentiment |
| `--summary` | off | LLM summary (implies --news, requires ollama) |
| `--earnings-summary` | off | LLM earnings/fundamentals summary |
| `--sector-analysis` | off | Sector ETF analysis with LLM |
| `--no-earnings` | off | Skip earnings in --summary (news-only) |
| `--email` | off | Send HTML email report |
| `--email-to ADDR` | env var | Override recipient address |
| `--data-path PATH` | none | CSV path for technical indicators |

---

### 6. Portfolio Filtering & Ranking

The portfolio ranker narrows the symbol universe before backtesting or live trading. Filters are composable.

```bash
# Rank by Sharpe ratio
python strategies/portfolio.py --data all_data_0.csv --top-k-sharpe 20 --summary

# Price band filter
python strategies/portfolio.py --data all_data_0.csv --price-above 50 --price-below 500 --summary

# Field filter (beta, rsi_14, pe_ratio, market_cap, etc.)
python strategies/portfolio.py --data all_data_0.csv --filter-field rsi_14 --filter-above 30 --filter-below 70 --summary

# Watchlist only
python strategies/portfolio.py --data all_data_0.csv --watchlist api_data/watchlist.csv --watchlist-mode filter --summary

# Combined pipeline — save for downstream use
python strategies/portfolio.py --data all_data_0.csv \
    --price-above 25 --filter-field beta --filter-below 1.5 --top-k-sharpe 15 \
    --output portfolio_picks.csv --summary
```

Output CSV has a `symbol` column — pipe into backtests with `--symbols-file portfolio_picks.csv`.

---

### 7. ML Training & Inference (Stockformer)

#### 7.1 Model Architecture

The StockTransformer is a transformer encoder for time-series classification/regression.

**9 models** = 3 label modes x 3 horizons:

| Label Mode | Horizons | Output | Loss |
|-----------|----------|--------|------|
| regression | 3d, 10d, 30d | Predicted return (float) | combined_regression |
| binary | 3d, 10d, 30d | Up/down classification | focal |
| buckets | 3d, 10d, 30d | Multi-class return buckets | soft_ordinal |

**Current trained model architectures:**

| Model Family | d_model | Layers | Feedforward | Heads |
|-------------|---------|--------|-------------|-------|
| Binary (bin_*) | 64 | 2 | 128 | 4 |
| Buckets (buck_*) | 64 | 2 | 128 | 4 |
| Regression (reg_*) | 128 | 3 | 256 | 4 |

**Architecture auto-detection:** At inference time, the model architecture (d_model, num_layers, dim_feedforward, nhead, num_buckets) is automatically read from the checkpoint state_dict. You never need to manually specify `--d-model` or `--num-layers` — just point at the model file and it works.

#### 7.2 Smoke Test

```bash
python -m stockformer.smoke_test --data-path "./all_data_*.csv"
```
Quick 1-epoch test to verify the pipeline works end to end.

#### 7.3 Train a Single Model

```bash
python -m stockformer.main --data-path "./all_data_*.csv" \
    --horizon 3 --label-mode binary --epochs 50 \
    --d-model 64 --num-layers 2 --dim-feedforward 128 \
    --train-end-date 2025-10-31 --infer-start-date 2025-11-01 \
    --model-out stockformer/output/models/model_bin_3d.pt \
    --output-csv stockformer/output/predictions/pred_bin_3d.csv
```

**Key training arguments:**
| Flag | Default | Description |
|------|---------|------------|
| `--data-path` | required | CSV glob pattern |
| `--horizon` | 3 | Prediction horizon: 3, 10, or 30 days |
| `--label-mode` | regression | regression, binary, or buckets |
| `--d-model` | 128 | Transformer embedding dimension |
| `--nhead` | 4 | Number of attention heads |
| `--num-layers` | 3 | Number of transformer encoder layers |
| `--dim-feedforward` | 256 | Feedforward layer size |
| `--batch-size` | 64 | Batch size |
| `--epochs` | 15 | Training epochs |
| `--lr` | default | Learning rate |
| `--dropout` | 0.1 | Dropout rate |
| `--loss-name` | auto | Loss function (focal, combined_regression, soft_ordinal, etc.) |
| `--train-end-date` | none | Training data cutoff |
| `--infer-start-date` | none | Inference data start |
| `--bucket-edges` | auto | Bucket boundaries (auto = quantile-based) |
| `--model-out` | auto | Model checkpoint path |
| `--output-csv` | auto | Prediction output path |
| `--save-config` | off | Save training config as JSON |

#### 7.4 Parallel Training (All 9 Models)

```bash
# Encoder architecture (default)
./scripts/run_parallel_training.sh

# Cross-attention architecture
./scripts/run_parallel_training.sh --cross-attention

# Custom epochs
./scripts/run_parallel_training.sh --epochs 30
```

Launches 3 parallel jobs (one per label mode), each training 3 horizons sequentially. ~12 hours total.

#### 7.5 Full Train + Infer + Merge Pipeline

```bash
# Weekly retrain — train all 9 models, then merge predictions
./scripts/train_infer_merge.sh

# With cross-attention
./scripts/train_infer_merge.sh --cross-attention
```

Monitor training:
```bash
tail -f logs/train_reg_encoder_*.log
tail -f logs/train_bin_encoder_*.log
tail -f logs/train_buck_encoder_*.log
```

#### 7.6 Inference Only (All 9 Models)

```bash
./scripts/infer_merge.sh                    # encoder (default)
./scripts/infer_merge.sh --cross-attention  # cross-attention
```

Runs inference on all 9 pre-trained models in parallel (~2 min), then merges into `merged_predictions.csv`. Architecture is auto-detected from checkpoints.

#### 7.7 Merge Predictions Manually

```bash
python -m stockformer.merge_predictions \
    --input-dir stockformer/output/predictions \
    --output merged_predictions.csv
```

---

### 8. Backtesting

#### 8.1 Bollinger Band Strategy

```bash
python strategies/run_bollinger.py \
    --data-path 'all_data_*.csv' \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --start-date 2025-08-01 --end-date 2026-01-31 \
    --output-trades trades.csv --output-signals signals/pending_orders.csv
```
- Entry: Price crosses below lower BB AND RSI <= 40
- Exit: Price reaches middle BB OR RSI > 70 OR hold >= 20 days

#### 8.2 Oversold Bounce Strategy

```bash
python strategies/run_oversold_bounce.py \
    --data-path 'all_data_*.csv' \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --start-date 2025-08-01 --end-date 2026-01-31
```
- Short hold (max 5 days), default instrument: option

#### 8.3 Oversold Reversal Strategy

```bash
python strategies/run_oversold_reversal.py \
    --data-path 'all_data_*.csv' \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --start-date 2025-08-01 --end-date 2026-01-31
```

#### 8.4 Low Bounce Strategy

```bash
python strategies/run_low_bounce.py \
    --data-path 'all_data_*.csv' \
    --watchlist api_data/watchlist.csv --watchlist-mode filter \
    --start-date 2025-08-01 --end-date 2026-01-31
```
- Longer hold (max 30 days), 52-week low bounce pattern

#### 8.5 ML Prediction Strategy

```bash
python strategies/run_backtest.py \
    --predictions merged_predictions.csv \
    --symbols AAPL GOOGL MSFT \
    --start-date 2025-08-01 --end-date 2026-01-31
```
- Entry: `pred_class = 1`, Exit: `pred_class = 0`

#### 8.6 Regression Momentum Strategy

```bash
python strategies/run_regression_momentum.py \
    --predictions merged_predictions.csv \
    --use-all-symbols \
    --start-date 2025-08-01 --end-date 2026-01-31
```
- Entry: `pred_reg_3d > 1%` AND `pred_reg_10d > 2%` AND `adx_signal > 0`
- Exit: `pred_reg_3d < 0` OR `cci_signal < 0` OR hold >= 13 days
- **Note:** Currently disabled in scan loop (missing 10d model, 0 trades)

#### 8.7 Pipeline Script

```bash
# Filter portfolio + run backtest in one command
./scripts/backtest_pipeline.sh --strategy bollinger \
    --start-date 2025-08-01 --end-date 2026-01-31
```

#### 8.8 Shared Strategy Arguments

All strategy runners inherit these from BaseRunner:

| Flag | Default | Description |
|------|---------|------------|
| `--mode` | backtest | backtest, daily, or live |
| `--data-path` | none | CSV glob pattern |
| `--predictions` | none | Merged predictions CSV |
| `--symbols` | none | Space-separated symbol list |
| `--symbols-file` | none | CSV file with symbols |
| `--use-all-symbols` | off | Use all symbols in data |
| `--watchlist` | none | Watchlist CSV path |
| `--watchlist-mode` | sort | sort or filter |
| `--start-date` | none | Backtest start (YYYY-MM-DD) |
| `--end-date` | none | Backtest end |
| `--start-cash` | 100000 | Initial capital |
| `--position-size` | 0.1 | Fraction per position (10%) |
| `--lookback-days` | 2 | Days for daily/live mode |
| `--no-notify` | off | Suppress email |
| `--summary-only` | off | Summary email only |
| `--skip-live` | off | Skip IBKR connection |
| `--instrument-type` | none | stock or option |
| `--output-trades` | none | Trade log CSV path |
| `--output-signals` | none | Signal CSV path |
| `--output-symbols` | none | Filtered symbol list path |
| `--output-rankings` | backtest_rankings.csv | Per-symbol rankings |

#### 8.9 Layered Exit Safety

All strategies use a two-layer exit system:

**Layer 1 (base_strategy.py):** Non-overridable backstop — `check_exit_safety()` checks SL/TP/trailing every evaluation.

**Layer 2 (each strategy):** Override class constants:

| Strategy | Stop Loss | Take Profit | Trailing | Activation | Max Hold |
|----------|-----------|-------------|----------|------------|----------|
| Base (default) | -8% | +15% | None | 0% | 20d |
| Oversold Bounce | -5% | +8% | -3% | +2% | 5d |
| Oversold Reversal | -5% | +10% | -4% | +2% | 7d |
| Bollinger | -7% | +12% | -5% | +3% | 20d |
| Low Bounce | -10% | +20% | -8% | +4% | 30d |
| Regression Momentum | -6% | +12% | -5% | +3% | 13d |

**Progressive trailing:** Trail narrows from full to half as price approaches take-profit.

#### 8.10 Backtest Output

Every backtest prints: per-symbol P&L, win/loss count, win rate, avg hold days, Sharpe ratio, portfolio summary.

Output files:
- `--output-trades trades.csv` — full trade log
- `--output-symbols symbols.csv` — filtered symbol list
- `--output-signals signals/pending_orders.csv` — signal CSV for execution

#### 8.11 Template Strategy (Build Your Own)

```bash
cp strategies/template_strategy.py strategies/my_strategy.py
# Edit: rename class, fill in _check_entry_conditions() and _check_exit_custom()
```

---

### 9. Live Execution (IBKR)

#### 9.1 Trading Mode Safety Gate

```
ibkr/trading_mode.conf:
  TRADING_MODE=paper    ← default (safe)
  TRADING_MODE=live     ← enables real money trading
```
The executor refuses to connect to live ports unless this file says `live`.

#### 9.2 Test Gateway Connectivity

```bash
python ibkr/test_gateway.py --balance --portfolio
python ibkr/test_gateway.py --test-notify   # test email notifications
```

#### 9.3 Execute Signals

```bash
# Dry run (preview)
python -m ibkr.execute_signals --signals signals/pending_orders.csv --dry-run

# Paper trading
python -m ibkr.execute_signals --signals signals/pending_orders.csv --gateway

# Live trading
python -m ibkr.execute_signals --signals signals/pending_orders.csv --live
```

#### 9.4 Signal CSV Format

```
action,symbol,shares,price,strategy,reason,timestamp,instrument_type
BUY,AAPL,0,227.50,bollinger,lower BB cross,2026-02-13,stock
BUY,CRWD,0,410.20,oversold_bounce,RSI bounce,2026-02-13,option
```
- `shares=0` → executor auto-sizes from account
- `price=0` → use live market price

#### 9.5 Watch Mode (Continuous)

```bash
python -m ibkr.execute_signals --watch --signals-dir signals/ --interval 30
```

#### 9.6 Close Positions

```bash
python -m ibkr.close_positions --all --gateway --client-id 10
python -m ibkr.close_positions --shares --gateway --client-id 10   # stocks only
python -m ibkr.close_positions --options --gateway --client-id 10  # options only
python -m ibkr.close_positions --all --gateway --dry-run           # preview
```

#### 9.7 Position Ledger

Active positions tracked in `signals/positions.json`:
```bash
python -c "
import json
with open('signals/positions.json') as f:
    data = json.load(f)
for key, pos in data.get('positions', {}).items():
    itype = pos.get('instrument_type', 'stock')
    print(f\"  {key:<30} {itype:<7} {pos['shares']:>4} @ \${pos['entry_price']:.2f}  \"
          f\"SL=\${pos.get('stop_price', 'N/A')}  TP=\${pos.get('take_profit_price', 'N/A')}  \"
          f\"strategy={pos['strategy']}  entry={pos['entry_date']}\")
print(f\"  Total: {len(data.get('positions', {}))} open position(s)\")
"
```

---

### 10. Cron Schedule (All Automation)

```bash
# ── EC2 crontab ──────────────────────────────────────────────────
# Nightly data pull + screener + web refresh (6pm ET)
0 18 * * 1-5 cd /home/sschmidt/bar_fly_trading && bash scripts/ec2_nightly.sh >> /var/log/bft/nightly.log 2>&1

# ── DGX crontab ──────────────────────────────────────────────────
# Inference + merge (7pm ET — after EC2 data pull)
0 19 * * 1-5 cd ~/proj/bar_fly_trading && bash scripts/infer_merge.sh >> logs/infer.log 2>&1

# Website LLM reports (7:30pm ET — after inference)
30 19 * * 1-5 cd ~/proj/bar_fly_trading && bash scripts/dgx_nightly.sh >> logs/dgx_nightly.log 2>&1
```

Real-time scan loop runs as a persistent background process (not cron):
```bash
# EC2
nohup bash scripts/rt_scan_loop.sh >> logs/rt_scan.log 2>&1 &
```

---

### 11. Reinforcement Learning (Experimental)

```bash
python bargyms/train.py
```
Train a PPO reinforcement learning agent on the custom trading environment.
