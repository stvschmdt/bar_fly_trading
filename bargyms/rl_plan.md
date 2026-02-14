# Config-Driven RL Trading Framework

## Context

The `bargyms/` package has 5 Gymnasium environments for RL trading, trained via Stable Baselines3 (PPO). The most mature (`GoldTradeEnv`) works but is brittle: hardcoded reward multipliers, magic-number thresholds, raw dollar prices (non-stationary), only ~30 of 96 available features, no connection to the `strategies/` backtest framework or `stockformer/` data pipeline.

**Goal**: Rebuild the core environment as a config-driven `TradingEnv` inside the existing `bargyms/` package structure (keeping the standard gymnasium `envs/`, `configs/`, `register.py` layout). It should:
- Consume the same `all_data_*.csv` data via `stockformer.data_utils`
- Reuse `BacktestAccount` / `StockOrder` / `check_exit_safety` from `strategies/`
- Support TD3, SAC, PPO via configurable action spaces (discrete, continuous)
- Drive all knobs (features, reward, episode structure, position sizing, exit safety) from YAML
- Bridge trained policies back into the live signal pipeline via a `PolicyStrategy(BaseStrategy)` adapter

**Why not wrap BaseStrategy directly?** BaseStrategy is built around rule-based `check_entry`/`check_exit` patterns. An RL agent makes a single action decision per step — it doesn't scan signals. Instead, the env uses `BacktestAccount` for execution and `check_exit_safety` for risk rails. Post-training, `PolicyStrategy` wraps the trained model into the `BaseStrategy` interface for signal generation.

---

## File Structure

Keep the existing `bargyms/` layout. New files marked with `+`, modified with `~`.

```
bargyms/
    __init__.py
    setup.py ~                          # Update deps: stable-baselines3, gymnasium
    register.py ~                       # Add TradingEnv-v0 registration
    train.py ~                          # Refactor: config-driven, multi-algo (PPO/SAC/TD3)
    inference.py ~                      # Refactor: use TradingEnv, date-range eval
    dataloader.py                       # Keep (legacy envs use it)

    bargyms/
        __init__.py
        envs/
            __init__.py ~               # Export TradingEnv
            analystgym.py               # Keep (legacy)
            binarygym.py                # Keep (legacy)
            benchmarkgym.py             # Keep (legacy)
            multiactiongym.py           # Keep (legacy)
            goldtradegym.py             # Keep (legacy)
            portfolioagent.py           # Keep (legacy)
            trading_env.py +            # NEW: Core config-driven environment
            action_spaces.py +          # NEW: Discrete / Continuous / MultiDiscrete handlers
            reward_functions.py +       # NEW: Pluggable reward registry
            observation_builder.py +    # NEW: Feature selection + account state construction

    configs/
        ppo_agent.yaml                  # Keep (legacy, works with GoldTradeEnv)
        ppo_discrete.yaml +             # NEW: PPO + Discrete(3) action space
        sac_continuous.yaml +           # NEW: SAC + Box(-1,1) continuous
        td3_continuous.yaml +           # NEW: TD3 + Box(-1,1) continuous

    policy_strategy.py +                # NEW: PolicyStrategy(BaseStrategy) adapter
    evaluate.py +                       # NEW: Backtest-equivalent eval via compute_stats
    run_rl_policy.py +                  # NEW: Runner script (like run_bollinger.py)
```

---

## YAML Config Schema

All environment/reward/algo knobs in one file. Example `configs/sac_continuous.yaml`:

```yaml
data:
  path: "all_data_*.csv"              # Glob pattern — uses stockformer.data_utils.load_panel_csvs()
  symbols: null                        # null = all symbols in data
  symbols_file: null                   # Or path to CSV with symbol list
  date_range:
    start: "2020-01-01"
    end: "2025-12-31"
  val_split: 0.2                       # Last 20% of dates for eval

features:
  preset: "standard"                   # "minimal" (15 cols) | "standard" (~35) | "full" (all 187)
  custom_columns: []                   # Extra columns to include
  exclude_columns: []                  # Columns to drop from preset
  market_context: true                 # Append SPY/sector features (from MARKET_FEATURE_COLUMNS)
  normalize: "zscore"                  # "zscore" | "minmax" | "none"

environment:
  lookback_window: 20                  # Days of history in observation
  episode_length: 40                   # Trading steps per episode
  symbols_per_episode: 1               # Single-symbol episodes
  action_space: "continuous"           # "discrete" | "continuous" | "multi_discrete"
  initial_cash: 100000
  position_size: 0.10                  # Max fraction per trade
  max_positions: 1
  allow_short: false
  transaction_cost_bps: 10             # 0.1% round-trip
  exit_safety:
    enabled: true
    stop_loss_pct: -0.08
    take_profit_pct: 0.15
    trailing_stop_pct: null
    trailing_activation_pct: 0.0
    max_hold_days: 20

reward:
  function: "sharpe_step"              # From reward registry
  params:
    sharpe_window: 20
    illegal_action_penalty: -1.0       # NOT -10000 like GoldTradeEnv
    holding_cost: 0.0
    no_trade_penalty: -0.001

algorithm:
  name: "SAC"                          # "PPO" | "SAC" | "TD3"
  policy: "MlpPolicy"
  policy_kwargs:
    net_arch:
      pi: [256, 256]
      qf: [256, 256]                   # SAC/TD3 critic
  hyperparameters:
    learning_rate: 0.0003
    gamma: 0.99
    batch_size: 256
    buffer_size: 100000
    tau: 0.005
    ent_coef: "auto"
    # PPO-only (ignored by SAC/TD3):
    n_steps: 2048
    n_epochs: 10
    clip_range: 0.2

training:
  total_timesteps: 1000000
  eval_freq: 10000
  n_eval_episodes: 20
  save_freq: 50000
  tensorboard_dir: "./rl_logs"
  model_save_path: "./rl_models"
  seed: 42
```

---

## Core Components

### 1. `bargyms/bargyms/envs/trading_env.py` — The Environment

```python
class TradingEnv(gym.Env):
    def __init__(self, data_loader, config, mode="train"):
        # Delegates
        self._action_handler = ActionSpaceFactory.create(config)
        self._reward_fn = RewardRegistry.get(config["reward"])
        self._obs_builder = ObservationBuilder(config)
        # Spaces
        self.observation_space = self._obs_builder.get_space()
        self.action_space = self._action_handler.get_space()

    def reset(self, seed=None, options=None):
        # Train: random symbol + random start date
        # Eval/inference: deterministic from options={symbol, start_date, end_date}
        # Create fresh position state (lightweight dict, not full BacktestAccount)
        # BacktestAccount only used in evaluate.py for full metrics

    def step(self, action):
        # 1. Decode action → (buy/sell/hold, size_fraction)
        # 2. Validate legality (can't sell if flat, can't buy if fully invested)
        # 3. Execute: update cash, shares, entry_price tracking
        # 4. Apply exit_safety: SL/TP/trailing (same math as BaseStrategy.check_exit_safety)
        # 5. Apply transaction costs
        # 6. Advance timestep
        # 7. Build observation via ObservationBuilder
        # 8. Compute reward via RewardFunction
        # 9. Check done (episode_length reached or account value < 50% initial)
        # Return (obs, reward, terminated, truncated, info)
```

**Position tracking during training** uses a lightweight internal dict (not full `BacktestAccount`) for speed — millions of steps need to be fast. The `info` dict at episode end includes a `trade_log` formatted identically to `BaseStrategy.trade_log` entries so `compute_stats()` works on it directly.

### 2. `bargyms/bargyms/envs/action_spaces.py` — Action Handlers

```python
class DiscreteActionHandler:
    # Discrete(3): 0=hold, 1=buy (fixed position_size fraction), 2=sell (close all)
    # Works with PPO

class ContinuousActionHandler:
    # Box(-1, 1, shape=(1,)): negative=sell, ~0=hold, positive=buy
    # Magnitude = fraction of available cash/position
    # Dead zone threshold (configurable, default ±0.1) prevents noise
    # Works with SAC, TD3, PPO

class MultiDiscreteActionHandler:
    # MultiDiscrete([3, 5]): [action_type, size_bucket]
    # 5 size buckets: 20%, 40%, 60%, 80%, 100% of position_size
    # Works with PPO
```

### 3. `bargyms/bargyms/envs/reward_functions.py` — Reward Registry

Available reward functions (all params from config, zero hardcoded numbers):

| Function | Signal | Best For |
|---|---|---|
| `pnl_step` | Dollar P&L change per step | Dense, simple baseline |
| `pnl_pct_step` | Percent portfolio change per step | Dense, normalized |
| `sharpe_step` | Rolling Sharpe of step returns | Risk-adjusted learning |
| `risk_adjusted` | Step return minus drawdown penalty | Drawdown-sensitive |
| `trade_completion` | Sparse: reward only on trade close | Clean credit assignment |

### 4. `bargyms/bargyms/envs/observation_builder.py` — Feature Construction

Observation shape: `(lookback_window, n_market_features + n_account_features)`

**Market features** from config preset (all stationary — matching stockformer convention):

| Preset | Columns | Examples |
|---|---|---|
| `minimal` | ~15 | rsi_14, adx_14, close_1d_roc, sma_20_pct, macd_signal, volume_pct, bull_bear_delta |
| `standard` | ~35 | Above + cci_14, atr_14, all signal columns, pe_ratio, beta, spy_ret, sector_ret |
| `full` | 187 | All `BASE_FEATURE_COLUMNS` from `stockformer/config.py` |

Column lists imported from `stockformer/config.py` (single source of truth, no duplication).

**Account features** (5, appended to each row):
- `cash_fraction` = cash / initial_cash
- `position_fraction` = position_value / portfolio_value
- `unrealized_pnl_pct` = (price - entry_price) / entry_price
- `days_held_normalized` = days_in_position / max_hold_days
- `is_long` = 1.0 if holding, 0.0 if flat

Per-ticker z-score normalization using training-period stats (same approach as `stockformer/dataset.py`). SB3's `VecNormalize` wrapper handles running normalization; stats saved alongside the trained model.

### 5. Data Loading — `RLDataLoader` (in `trading_env.py` or separate)

```python
class RLDataLoader:
    def __init__(self, config):
        # Uses stockformer.data_utils.load_panel_csvs(config.data.path)
        # Filters date range, selects feature columns from preset
        # Per-ticker z-score normalization
        # Stores as {symbol: np.ndarray} for fast env access

    def get_symbol_data(self, symbol) -> np.ndarray  # (T, F)
    def get_prices(self, symbol) -> np.ndarray         # (T,) adjusted_close
    def get_dates(self, symbol) -> np.ndarray           # (T,) dates
    def get_symbols(self) -> list[str]
```

### 6. `bargyms/train.py` — Multi-Algorithm Training

```python
ALGO_MAP = {"PPO": sb3.PPO, "SAC": sb3.SAC, "TD3": sb3.TD3}

# Usage:
#   python bargyms/train.py --config bargyms/configs/sac_continuous.yaml
#   python bargyms/train.py --config bargyms/configs/ppo_discrete.yaml --symbols NVDA AAPL
```

Config validation: SAC/TD3 require `action_space: "continuous"`, error otherwise.

### 7. `bargyms/policy_strategy.py` — Live Trading Bridge

```python
class PolicyStrategy(BaseStrategy):
    """Wraps trained SB3 model as a BaseStrategy for signal generation."""
    STRATEGY_NAME = "rl_policy"

    def __init__(self, account, symbols, model_path, config_path, ...):
        # Load SB3 model, create RLDataLoader + ObservationBuilder from config
        # Set exit safety params from config

    def evaluate(self, date, current_prices, options_data) -> list[Order]:
        # For each symbol:
        #   1. Build observation (lookback window ending at date)
        #   2. model.predict(obs, deterministic=True)
        #   3. Decode action → BUY/SELL/HOLD
        #   4. Emit StockOrder if buy or sell

    def check_entry(self, row): return False  # Not used — evaluate() handles all
    def fetch_realtime(self, symbol): return pd.DataFrame()  # Uses overnight data
```

This plugs into the existing signal pipeline:
- `run_rl_policy.py` (like `run_bollinger.py`) creates `PolicyStrategy` + `BaseRunner`
- Signals written to `signals/pending_orders.csv` via `SignalWriter`
- Executed via `ibkr/execute_signals.py` → IBKR Gateway

### 8. `bargyms/evaluate.py` — Metrics Bridge

Runs trained model through rolling episodes over the full eval date range, collects `trade_log` entries, feeds to `strategies/backtest_stats.compute_stats()` — produces the exact same metrics table (win rate, Sharpe, per-symbol P&L, avg hold days) as rule-based strategy backtests.

---

## Exit Safety Integration

The env reuses the exact same math from `BaseStrategy.check_exit_safety()`:
- Hard stop-loss and take-profit checks each step
- Trailing stop with progressive tightening (same formula)
- Max hold days force-close
- All thresholds from config, not hardcoded

This ensures the RL agent trains under the same risk constraints as rule-based strategies, making metrics directly comparable.

---

## Key Reuse Points

| What | From | Used In |
|---|---|---|
| `load_panel_csvs()` | `stockformer/data_utils.py` | `RLDataLoader` data loading |
| `BASE_FEATURE_COLUMNS` | `stockformer/config.py` | Feature preset definitions |
| `MARKET_FEATURE_COLUMNS` | `stockformer/config.py` | Market context features |
| `BacktestAccount` | `account/backtest_account.py` | `evaluate.py` full metrics |
| `StockOrder`, `OrderOperation` | `order.py` | `PolicyStrategy.evaluate()` |
| `check_exit_safety` math | `strategies/base_strategy.py` | `TradingEnv.step()` safety rails |
| `compute_stats()` | `strategies/backtest_stats.py` | `evaluate.py` metrics |
| `SignalWriter` | `strategies/signal_writer.py` | `run_rl_policy.py` signal output |
| `BaseStrategy` / `BaseRunner` | `strategies/` | `PolicyStrategy` adapter |

---

## Implementation Order

| Phase | Files | What |
|---|---|---|
| **1: Core env** | `observation_builder.py`, `action_spaces.py`, `reward_functions.py`, `trading_env.py` | Environment that can step through data with configurable obs/action/reward |
| **2: Training** | `train.py` (refactor), `configs/*.yaml`, `register.py` (update) | Train PPO/SAC/TD3 from YAML config |
| **3: Eval + Bridge** | `evaluate.py`, `policy_strategy.py`, `run_rl_policy.py`, `inference.py` (refactor) | Metrics, live signal generation, backtest comparison |

## Verification

1. `python bargyms/train.py --config bargyms/configs/ppo_discrete.yaml` — trains, TensorBoard shows reward curve
2. `python bargyms/train.py --config bargyms/configs/sac_continuous.yaml` — SAC trains with continuous actions
3. `python bargyms/evaluate.py --model rl_models/best_model.zip --config configs/sac_continuous.yaml` — produces same metrics format as `strategies/backtest_stats.py`
4. `python bargyms/run_rl_policy.py --model-path ... --config-path ... --mode backtest` — runs through `backtest.py`, comparable to `run_bollinger.py`
5. Legacy envs still work: `python bargyms/train.py` with old `ppo_agent.yaml` unchanged
