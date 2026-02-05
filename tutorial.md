# bar_fly_trading - End to End Tutorial

## Setup

1a.
```bash
docker pull mysql:latest && docker run --name mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -e MYSQL_DATABASE=bar_fly_trading -p 3306:3306 -d mysql:latest
```
# Option A: Start a local MySQL database in Docker

1b.
```bash
# Ubuntu/Debian
sudo apt install mysql-server
sudo systemctl start mysql
sudo mysql -u root -e "CREATE DATABASE bar_fly_trading; ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'my-secret-pw'; FLUSH PRIVILEGES;"

# macOS
brew install mysql && brew services start mysql
mysql -u root -e "CREATE DATABASE bar_fly_trading; ALTER USER 'root'@'localhost' IDENTIFIED BY 'my-secret-pw'; FLUSH PRIVILEGES;"
```
# Option B: Install MySQL natively without Docker

2.
```bash
export MYSQL_PASSWORD=my-secret-pw
export ALPHAVANTAGE_API_KEY=your_key_here
```
# Set required environment variables for DB access and Alpha Vantage API

3.
```bash
pip install -r requirements.txt
```
# Install Python dependencies

## Data Pipeline

4.
```bash
python -m api_data.pull_api_data -w all
```
# Pull OHLCV, technical indicators, fundamentals, options, and economic data for all S&P500 + watchlist symbols into the DB

5.
```bash
python -m api_data.pull_api_data -s AAPL NVDA MSFT
```
# Pull data for specific symbols only (faster for testing)

6.
```bash
python -m api_data.pull_api_data --gold-table-only
```
# Rebuild the gold table (joins all features into one table) without re-fetching from the API

7.
```bash
python api_data/backfill_options.py -s AAPL NVDA --start_date 2024-01-01 --end_date 2025-01-01
```
# Backfill historical options data for specific symbols and date range (multi-threaded)

## Screening & Visualization

8.
```bash
python -m visualizations.screener --n_days 60 --data .
```
# Run the stock screener on all symbols, generate technical charts and an overnight PDF report

9.
```bash
python -m visualizations.screener --symbols AAPL NVDA --n_days 30 --data .
```
# Run the screener for specific symbols only

## Nightly Cron (Automated Pipeline)

10.
```bash
python cron.py
```
# Run the full nightly pipeline: pull all data, run screener, generate overnight PDF, upload to Google Drive

## Real-Time Utilities

11.
```bash
python -m api_data.rt_utils AAPL
```
# Fetch real-time quote and nearest monthly options chain for a symbol

12.
```bash
python -m api_data.rt_utils AAPL --news
```
# Add news sentiment analysis with weighted scoring across top 20 articles

13.
```bash
python -m api_data.rt_utils AAPL --earnings-summary
```
# Fetch earnings and company overview, then generate a local LLM summary of fundamentals

14.
```bash
python -m api_data.rt_utils AAPL --news --summary
```
# Full report: quote, options, news sentiment, earnings overview, and combined LLM summary

15.
```bash
python -m api_data.rt_utils AAPL --news --summary --no-earnings
```
# News-only LLM summary (skip earnings API calls, useful for daily runs after initial review)

16.
```bash
python -m api_data.rt_utils AAPL --news --summary --email
```
# Run full report and send formatted HTML email to the configured recipient list

17.
```bash
python -m api_data.rt_utils AAPL --months-out 2 --strikes 3 --type call
```
# Customize options display: 2 months out expiration, 3 strikes each side of ATM, calls only

## ML Model Training (Stockformer)

18.
```bash
python -m stockformer.smoke_test --data-path "./all_data_*.csv"
```
# Quick 1-epoch smoke test to verify the training pipeline works end to end

19.
```bash
python -m stockformer.main --data-path "./all_data_*.csv"
```
# Train all 9 models (3 horizons x 3 label modes: regression, binary, buckets)

20.
```bash
python -m stockformer.main --data-path "./all_data_*.csv" --horizon 3 --label-mode binary --num-epochs 50
```
# Train a single model: 3-day horizon, binary classification

21.
```bash
./scripts/run_parallel_training.sh --cross-attention
```
# Train all 9 models in parallel (3 label modes run concurrently), reduces ~36h to ~12h

22.
```bash
python -m stockformer.main --data-path "./all_data_*.csv" --horizon 3 --label-mode regression --infer-only --model-out model.pt
```
# Run inference only on a trained model (skip training)

23.
```bash
python -m stockformer.merge_predictions --input-dir stockformer/output --output merged_predictions.csv
```
# Merge all 9 prediction CSVs into a single file with all prediction columns

## Backtesting

24.
```bash
python backtest.py --strategy_name BollingerBandsStrategy --start_date 2024-01-01 --end_date 2024-12-31 --symbols AAPL NVDA
```
# Backtest the Bollinger Bands strategy on historical data from the DB

25.
```bash
python strategies/run_regression_momentum.py --use-all-symbols --start-date 2024-07-01 --end-date 2024-12-31
```
# Backtest the regression momentum strategy using stockformer predictions across all symbols

26.
```bash
python strategies/run_regression_momentum.py --symbols AAPL GOOGL MSFT --start-date 2024-07-01 --end-date 2024-12-31 --start-cash 100000
```
# Backtest regression momentum for specific symbols with $100k starting cash

## Live Trading (IBKR)

27.
```bash
python ibkr/test_gateway.py --balance --portfolio
```
# Test IBKR Gateway connectivity and view account balance and positions

28.
```bash
python ibkr/run_live_strategy.py --symbols AAPL NVDA MSFT
```
# Start paper trading the regression momentum strategy with IBKR (evaluates every 5 min)

29.
```bash
python ibkr/run_live_strategy.py --symbols AAPL NVDA MSFT --live --once
```
# Execute one live trading evaluation and exit (for cron-based scheduling)

## Reinforcement Learning (Experimental)

30.
```bash
python bargyms/train.py
```
# Train a PPO reinforcement learning agent on the custom trading environment

## Data Validation

31.
```bash
python api_data/validate_db.py
```
# Validate data integrity in the MySQL database (check for duplicates, missing values)

32.
```bash
python api_data/validate.py -c all_data.csv
```
# Validate a CSV data file for quality issues

## Email Notifications

33.
```bash
export IBKR_SMTP_SERVER=smtp.gmail.com
export IBKR_SMTP_USER=you@gmail.com
export IBKR_SMTP_PASSWORD=your_app_password
export IBKR_NOTIFY_EMAIL=recipient1@gmail.com,recipient2@gmail.com
```
# Configure email for rt_utils reports and IBKR trade notifications (comma-separated recipients)

## Typical Daily Workflow

34.
```bash
python cron.py
```
# Morning: run nightly pipeline to refresh data and generate overnight PDF

35.
```bash
python -m api_data.rt_utils AAPL --news --summary --email
```
# During market hours: check real-time quote, news sentiment, and LLM analysis, send to email

36.
```bash
python ibkr/run_live_strategy.py --symbols AAPL NVDA MSFT --once
```
# Execute one round of live strategy signals based on latest predictions