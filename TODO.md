# Trading System TODO

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DAILY TRADING PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Data Pull   │────▶│   ML Train   │────▶│  Inference   │
  │  (api_data)  │     │  (weekly)    │     │  (daily)     │
  └──────────────┘     └──────────────┘     └──────────────┘
         │                                         │
         │                                         ▼
         │                              ┌──────────────────┐
         │                              │ Predictions CSV  │
         │                              │ (3d, 10d, 30d)   │
         │                              └────────┬─────────┘
         │                                       │
         ▼                                       ▼
  ┌──────────────┐                    ┌──────────────────┐
  │  all_data    │───────────────────▶│    STRATEGIES    │
  │  (features)  │                    ├──────────────────┤
  └──────────────┘                    │ Bollinger Bands  │
                                      │ Regression Mom.  │
                                      │ (future: more)   │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  Signal Filter   │
                                      │  & Ranking       │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  Pre-Trade Check │
                                      │  - Real-time quote│
                                      │  - Spread check  │
                                      │  - Cash/position │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  Trade Execution │
                                      │  (IB Gateway)    │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  Notifications   │
                                      │  Email / SMS     │
                                      └──────────────────┘
```

---

## Phase 1: Foundation & Testing (Current)

### 1.1 ML Pipeline Integration
- [ ] Add `--train` flag to stockformer to conditionally retrain
- [ ] Create wrapper script: `run_daily_pipeline.sh`
  - Pulls latest data
  - Runs inference (daily)
  - Runs training (weekly, e.g., Sundays)
- [ ] Ensure stockformer outputs to predictable paths for downstream

### 1.2 Strategy Testing
- [ ] Test Bollinger shadow strategy with real data
- [ ] Verify email notifications arrive correctly
- [ ] Document false positive rate (signals that shouldn't trade)

### 1.3 Regression Momentum Strategy
- [ ] Port/create `regression_momentum_strategy.py` for ibkr
- [ ] Integrate stockformer predictions (3d, 10d, 30d)
- [ ] Shadow deploy alongside Bollinger strategy

---

## Phase 2: Signal Processing

### 2.1 Post-Processing / Filtering Script
- [ ] Create `signal_ranker.py` to filter and prioritize signals
- [ ] Ranking criteria:
  - Signal strength (distance from band, prediction confidence)
  - Volume (avoid illiquid stocks)
  - Sector diversification (don't over-concentrate)
  - Recent performance (avoid catching falling knives)
  - Max signals per day (e.g., top 3-5)
- [ ] Output: prioritized list with "execute" vs "watch" tags

### 2.2 Signal Combination
- [ ] Combine Bollinger + ML signals for higher confidence
- [ ] Example: Only trade if Bollinger BUY + ML predicts positive 3d return
- [ ] Add configurable thresholds

---

## Phase 3: Real-Time Data

### 3.1 Alpha Vantage Integration
- [ ] Add Alpha Vantage API client for real-time quotes
- [ ] Pre-execution quote check:
  - Verify current price vs signal price (stale data check)
  - Check bid/ask spread (avoid wide spreads)
  - Confirm volume is reasonable
- [ ] Environment variable: `ALPHAVANTAGE_API_KEY`

### 3.2 Price Drift Protection
- [ ] Reject trade if price moved >X% since signal
- [ ] Configurable threshold (e.g., 2%)
- [ ] Log rejected trades for analysis

---

## Phase 4: Live Trading (Small Scale)

### 4.1 Paper Trading First
- [ ] Run on paper account (port 4002) for 1-2 weeks
- [ ] Compare shadow signals vs paper executions
- [ ] Track slippage and fill rates

### 4.2 Small Live Test
- [ ] Start with 1 share per trade
- [ ] Single strategy (Bollinger or ML, not both)
- [ ] Limited universe (e.g., 10 liquid stocks)
- [ ] Manual approval gate (email confirmation before execute?)

### 4.3 Position Limits
- [ ] Max position per stock (e.g., 5% of portfolio)
- [ ] Max total positions (e.g., 10)
- [ ] Daily loss limit (halt trading if exceeded)

---

## Phase 5: Hardening & Robustness

### 5.1 Error Handling
- [ ] Retry logic for transient failures (network, API)
- [ ] Graceful degradation (continue with available data)
- [ ] Alert on errors (email/SMS for failures)
- [ ] Dead man's switch (alert if pipeline doesn't run)

### 5.2 Logging & Monitoring
- [ ] Structured logging (JSON for parsing)
- [ ] Daily summary report (trades, P&L, errors)
- [ ] Dashboard (simple HTML or Grafana)

### 5.3 Data Validation
- [ ] Sanity checks on input data (missing columns, NaN)
- [ ] Price reasonableness checks (no $0 or $999999)
- [ ] Date freshness checks (reject stale data)

### 5.4 Testing
- [ ] Unit tests for signal generation
- [ ] Integration tests with mock Gateway
- [ ] Backtest validation (signals match historical)

---

## Phase 6: Options Trading

### 6.1 Options Signal Generation
- [ ] Extend strategies to output options signals
- [ ] Bull call spread on strong BUY
- [ ] Bear put spread on strong SELL
- [ ] Define entry criteria (IV, DTE, delta)

### 6.2 Options Pre-Trade Checks
- [ ] Real-time options chain (via IBKR or Alpha Vantage)
- [ ] Spread check (bid-ask width)
- [ ] Open interest check (liquidity)
- [ ] IV percentile check (avoid expensive options)

### 6.3 Options Execution Safety
- [ ] Limit orders only (no market orders on options)
- [ ] Max premium per contract
- [ ] Position sizing based on max loss
- [ ] Auto-close at profit target or expiry threshold

---

## Environment Variables Needed

```bash
# Existing
IBKR_SMTP_SERVER, IBKR_SMTP_USER, IBKR_SMTP_PASSWORD, IBKR_NOTIFY_EMAIL

# To Add
ALPHAVANTAGE_API_KEY="your-key"
IBKR_MAX_DAILY_LOSS="1000"
IBKR_MAX_POSITION_PCT="0.05"
IBKR_ENABLE_LIVE_TRADING="false"  # safety flag
```

---

## File Structure (Proposed)

```
bar_fly_trading/
├── ibkr/
│   ├── bollinger_shadow_strategy.py   # ✅ Done
│   ├── regression_momentum_strategy.py # TODO
│   ├── signal_ranker.py               # TODO
│   ├── realtime_quotes.py             # TODO (Alpha Vantage)
│   ├── options_strategy.py            # TODO
│   └── daily_pipeline.py              # TODO (orchestrator)
├── stockformer/                        # ✅ Done
│   └── (ML training & inference)
├── scripts/
│   ├── run_daily_pipeline.sh          # TODO
│   └── run_weekly_train.sh            # TODO
└── TODO.md                             # ✅ This file
```

---

## Priority Order

| Priority | Task | Effort | Risk |
|----------|------|--------|------|
| P0 | Test Bollinger shadow with real data | Low | None |
| P0 | Verify notifications working | Low | None |
| P1 | Regression momentum strategy | Medium | None |
| P1 | Signal ranker/filter | Medium | None |
| P2 | Alpha Vantage real-time quotes | Medium | None |
| P2 | Paper trading test | Low | None |
| P3 | Small live trading test | Low | Medium |
| P3 | Error handling & logging | Medium | None |
| P4 | Options trading | High | High |

---

## Next Immediate Steps

1. **Tomorrow**: Run `bollinger_shadow_strategy.py` with real all_data.csv
2. **This Week**: Create `regression_momentum_strategy.py`
3. **This Week**: Build `signal_ranker.py` for filtering
4. **Next Week**: Add Alpha Vantage for real-time quotes
5. **Week After**: Paper trading pilot

---

## Notes

- All strategies are SHADOW mode by default until explicitly enabled
- Every trade requires Gateway connection for real account data
- Email notifications for all signals (shadow and live)
- Start conservative: 1 share, few stocks, tight stops


# 1. On the source server, dump the database
mysqldump -u root -p bar_fly_trading > bar_fly_trading_dump.sql

# 2. Copy the dump file to this machine
scp bar_fly_trading_dump.sql stvschmdt@<this-machine>:~/

# 3. Then on this machine, import it
docker exec -i mysql mysql -u root -pmy-secret-pw bar_fly_trading < ~/bar_fly_trading_dump.sql

