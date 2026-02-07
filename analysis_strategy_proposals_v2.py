"""
Deep Dive: StockFormer v2 Model Performance & Strategy Proposals
================================================================
Analyzes all 9 merged v2 models (3 label_modes x 3 horizons) +
technical/signal data to propose trading strategies.

v2 improvements over v1:
  - 46 features (was 8): technicals, signals, fundamentals, macro
  - Positional encoding + attention pooling + 2-layer MLP heads
  - Temporal train/val split (no look-ahead bias)
  - Training-only normalization stats
  - AdamW + cosine LR scheduler + warmup + gradient clipping
  - Early stopping with best-model checkpointing
  - Lookback = 20 (was 5)

Output: analysis_strategy_proposals_v2.txt
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import OrderedDict

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 140)
pd.set_option('display.float_format', '{:.4f}'.format)

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('merged_predictions_v2.csv')
df['date'] = pd.to_datetime(df['date'])

lines = []
def pr(s=''):
    lines.append(s)
    print(s)

def header(title, char='═'):
    pr()
    pr(char * 80)
    pr(f'  {title}')
    pr(char * 80)

def subheader(title, char='─'):
    pr()
    pr(f'  {char * 3} {title} {char * 3}')

def safe_sharpe(rets, horizon):
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return rets.mean() / rets.std() * np.sqrt(252 / horizon)

def strategy_stats(rets, horizon, big_threshold=None):
    """Compute standard strategy statistics."""
    if big_threshold is None:
        big_threshold = {3: 3, 10: 5, 30: 10}.get(horizon, 5)
    n = len(rets)
    if n == 0:
        return None
    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    wr = (rets > 0).mean()
    avg = rets.mean()
    med = rets.median()
    sharpe = safe_sharpe(rets, horizon)
    big_w = (rets > big_threshold).sum()
    big_l = (rets < -big_threshold).sum()
    return {
        'trades': n, 'wins': wins, 'losses': losses,
        'win_rate': wr, 'avg_ret': avg, 'median_ret': med,
        'sharpe': sharpe, 'best': rets.max(), 'worst': rets.min(),
        'big_wins': big_w, 'big_losses': big_l,
        'big_threshold': big_threshold,
    }

def print_strategy(stats, horizon):
    if stats is None:
        pr('  No trades triggered')
        return
    bt = stats['big_threshold']
    pr(f'  Trades: {stats["trades"]:,} | Win rate: {stats["win_rate"]:.1%}')
    pr(f'  Avg return: {stats["avg_ret"]:.2f}% | Median: {stats["median_ret"]:.2f}%')
    pr(f'  Best: {stats["best"]:.1f}% | Worst: {stats["worst"]:.1f}%')
    pr(f'  Wins: {stats["wins"]} | Losses: {stats["losses"]} | Big wins (>{bt}%): {stats["big_wins"]} | Big losses (<-{bt}%): {stats["big_losses"]}')
    pr(f'  Sharpe (annualized): {stats["sharpe"]:.2f}')


# ══════════════════════════════════════════════════════════════════════
header('STOCKFORMER v2 — FULL MODEL PERFORMANCE & STRATEGY REPORT')
# ══════════════════════════════════════════════════════════════════════

pr(f'\n  Data: {len(df):,} rows | {df.ticker.nunique()} tickers | {df.date.min().date()} to {df.date.max().date()}')
pr(f'  Models: 9 (3 label_modes x 3 horizons)')
pr(f'  Architecture: Transformer + PositionalEncoding + AttentionPooling + MLP heads')
pr(f'  Features: 46 columns | Lookback: 20 days | Optimizer: AdamW + cosine LR')
pr(f'  Train/Val: Temporal split (train < 2023-11-13, val >= 2023-11-13)')

# ──────────────────────────────────────────────────────────────────────
# PART 1: ALL 9 MODEL QUALITY REPORTS
# ──────────────────────────────────────────────────────────────────────
header('PART 1: MODEL QUALITY — ALL 9 v2 MODELS')

# --- Regression models ---
subheader('Regression Models (predict raw % return)')
for horizon, tag in [(3, 'reg_3d'), (10, 'reg_10d'), (30, 'reg_30d')]:
    pred_col = f'pred_return_{tag}'
    true_col = f'true_return_{tag}'
    if pred_col not in df.columns:
        continue
    p, t = df[pred_col].dropna(), df[true_col].dropna()
    common = p.index.intersection(t.index)
    p, t = p[common], t[common]

    corr = p.corr(t)
    mae = (p - t).abs().mean()
    rmse = np.sqrt(((p - t) ** 2).mean())
    dir_acc = ((p > 0) == (t > 0)).mean()

    up_mask = p > 0
    up_acc = (t[up_mask] > 0).mean() if up_mask.sum() > 0 else 0
    up_avg_ret = t[up_mask].mean() if up_mask.sum() > 0 else 0
    down_mask = p < 0
    down_acc = (t[down_mask] < 0).mean() if down_mask.sum() > 0 else 0
    down_avg_ret = t[down_mask].mean() if down_mask.sum() > 0 else 0

    q80, q20 = p.quantile(0.8), p.quantile(0.2)
    top_q = t[p >= q80]
    bot_q = t[p <= q20]

    pr(f'\n  {tag.upper()} ({horizon}-day horizon)')
    pr(f'    Correlation:      {corr:.4f}')
    pr(f'    MAE:              {mae:.4f}  |  RMSE: {rmse:.4f}')
    pr(f'    Direction Acc:    {dir_acc:.1%}')
    pr(f'    Pred range:       [{p.min():.3f}, {p.max():.3f}] (std={p.std():.4f})')
    pr(f'    "Up" calls:       {up_mask.sum():,} ({up_acc:.1%} correct, avg true ret {up_avg_ret:.3f})')
    pr(f'    "Down" calls:     {down_mask.sum():,} ({down_acc:.1%} correct, avg true ret {down_avg_ret:.3f})')
    pr(f'    Top quintile:     avg true return {top_q.mean():.3f} (n={len(top_q):,})')
    pr(f'    Bottom quintile:  avg true return {bot_q.mean():.3f} (n={len(bot_q):,})')
    pr(f'    Top-Bottom spread: {top_q.mean() - bot_q.mean():.3f}')

# --- Binary models ---
subheader('Binary Models (predict up/down class)')
for horizon, tag in [(3, 'bin_3d'), (10, 'bin_10d'), (30, 'bin_30d')]:
    pred_col = f'pred_class_{tag}'
    true_col = f'true_class_{tag}'
    prob_col = f'prob_1_{tag}'
    reg_tag = tag.replace('bin_', 'reg_')
    true_ret_col = f'true_return_{reg_tag}'

    if pred_col not in df.columns:
        continue

    p, t = df[pred_col], df[true_col]
    prob = df[prob_col]
    true_ret = df.get(true_ret_col)

    acc = (p == t).mean()
    tp = ((p == 1) & (t == 1)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()
    tn = ((p == 0) & (t == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    high_bull = prob > 0.6
    high_bear = prob < 0.4
    bull_correct = (t[high_bull] == 1).mean() if high_bull.sum() > 0 else 0
    bear_correct = (t[high_bear] == 0).mean() if high_bear.sum() > 0 else 0
    bull_avg_ret = true_ret[high_bull].mean() if high_bull.sum() > 0 and true_ret is not None else 0
    bear_avg_ret = true_ret[high_bear].mean() if high_bear.sum() > 0 and true_ret is not None else 0

    pr(f'\n  {tag.upper()} ({horizon}-day horizon)')
    pr(f'    Accuracy:         {acc:.1%}')
    pr(f'    Class dist:       pred_up={int((p==1).sum()):,}  pred_down={int((p==0).sum()):,}')
    pr(f'    Precision (up):   {precision:.1%}  |  Recall (up): {recall:.1%}  |  Specificity: {spec:.1%}')
    pr(f'    Prob distribution: mean={prob.mean():.3f} std={prob.std():.3f} [min={prob.min():.3f}, max={prob.max():.3f}]')
    pr(f'    High-conf bull (p>0.6):  {high_bull.sum():,} trades, {bull_correct:.1%} correct, avg ret {bull_avg_ret:.3f}')
    pr(f'    High-conf bear (p<0.4):  {high_bear.sum():,} trades, {bear_correct:.1%} correct, avg ret {bear_avg_ret:.3f}')

# --- Bucket models ---
subheader('Bucket Models (8-class distribution)')
bucket_edges = [-6, -4, -2, 0, 2, 4, 6]
bucket_labels = ['< -6%', '-6 to -4%', '-4 to -2%', '-2 to 0%',
                 '0 to 2%', '2 to 4%', '4 to 6%', '> 6%']

for horizon, tag in [(3, 'buck_3d'), (10, 'buck_10d'), (30, 'buck_30d')]:
    pred_col = f'pred_class_{tag}'
    true_col = f'true_class_{tag}'
    if pred_col not in df.columns:
        continue
    p, t = df[pred_col], df[true_col]
    acc = (p == t).mean()

    pr(f'\n  {tag.upper()} ({horizon}-day, 8 buckets)')
    pr(f'    Overall accuracy: {acc:.1%} (random baseline: 12.5%)')
    pr(f'    Predicted class dist: {dict(sorted(p.value_counts().to_dict().items()))}')
    pr(f'    Per-bucket accuracy:')
    for b in range(8):
        mask = t == b
        if mask.sum() > 0:
            b_acc = (p[mask] == b).mean()
            b_n = mask.sum()
            pred_b = (p == b).sum()
            pr(f'      Bucket {b} ({bucket_labels[b]:>12s}): {b_acc:.1%} hit (true={b_n:,}, pred={pred_b:,})')

    # Expected return from bucket probabilities
    reg_tag = tag.replace('buck_', 'reg_')
    exp_ret_col = f'pred_expected_return_{tag}'
    true_ret_col = f'true_return_{reg_tag}'
    if exp_ret_col in df.columns and true_ret_col in df.columns:
        exp_ret = df[exp_ret_col]
        true_ret = df[true_ret_col]
        corr = exp_ret.corr(true_ret)
        pr(f'    Expected return correlation with true: {corr:.4f}')


# ──────────────────────────────────────────────────────────────────────
# PART 2: SIGNAL ENRICHMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────
header('PART 2: TECHNICAL SIGNAL ENRICHMENT')

signal_cols = ['sma_cross_signal', 'bollinger_bands_signal', 'rsi_signal', 'macd_signal',
               'adx_signal', 'cci_signal', 'atr_signal', 'pe_ratio_signal']

subheader('Individual Signal Win Rates')
for horizon_days, ret_col in [(3, 'true_return_reg_3d'), (10, 'true_return_reg_10d'), (30, 'true_return_reg_30d')]:
    pr(f'\n  {horizon_days}-day forward returns:')
    pr(f'  {"Signal":<25s} {"Bullish(1)":<22s} {"Neutral(0)":<22s} {"Bearish(-1)":<22s}')
    pr(f'  {"─"*25} {"─"*22} {"─"*22} {"─"*22}')
    if ret_col not in df.columns:
        continue
    for sig in signal_cols:
        if sig not in df.columns:
            continue
        parts = []
        for val in [1, 0, -1]:
            mask = df[sig] == val
            if mask.sum() > 50:
                avg_ret = df.loc[mask, ret_col].mean()
                win_rate = (df.loc[mask, ret_col] > 0).mean()
                parts.append(f'{win_rate:.0%} / {avg_ret:+.3f} (n={mask.sum()})')
            else:
                parts.append(f'n/a')
        pr(f'  {sig:<25s} {parts[0]:<22s} {parts[1]:<22s} {parts[2]:<22s}')

# Bull bear delta analysis
subheader('Bull-Bear Delta Composite Signal')
for horizon, col in [(3, 'true_return_reg_3d'), (10, 'true_return_reg_10d'), (30, 'true_return_reg_30d')]:
    if col not in df.columns:
        continue
    pr(f'\n  {horizon}-day forward returns by bull_bear_delta:')
    pr(f'  {"Delta":<8s} {"Count":<8s} {"Win Rate":<10s} {"Avg Ret":<12s} {"Med Ret":<12s} {"Sharpe":<10s}')
    for delta in sorted(df['bull_bear_delta'].unique()):
        mask = df['bull_bear_delta'] == delta
        if mask.sum() < 30:
            continue
        rets = df.loc[mask, col]
        wr = (rets > 0).mean()
        avg = rets.mean()
        med = rets.median()
        sharpe = safe_sharpe(rets, horizon)
        pr(f'  {delta:>+5.0f}   {mask.sum():<8d} {wr:<10.1%} {avg:<+12.4f} {med:<+12.4f} {sharpe:<10.2f}')


# ──────────────────────────────────────────────────────────────────────
# PART 3: STRATEGY PROPOSALS (all 9 models available)
# ──────────────────────────────────────────────────────────────────────
header('PART 3: STRATEGY PROPOSALS')

strategies = OrderedDict()

# Strategy 1: Quick Momentum Consensus (3-day)
subheader('Strategy 1: "Quick Momentum Consensus" (3-day hold)')
pr('  Rule: bin_3d=UP + reg_3d > 0.05 + bull_bear_delta >= 2')
mask = ((df['pred_class_bin_3d'] == 1) &
        (df['pred_return_reg_3d'] > 0.05) &
        (df['bull_bear_delta'] >= 2))
rets = df.loc[mask, 'true_return_reg_3d']
stats = strategy_stats(rets, 3)
print_strategy(stats, 3)
if stats: strategies['Quick Momentum'] = stats

# Strategy 2: 10-Day High-Confidence
subheader('Strategy 2: "10-Day High-Confidence" (10-day hold)')
pr('  Rule: prob_1_bin_10d > 0.55 + reg_10d > 0.3')
mask = ((df['prob_1_bin_10d'] > 0.55) &
        (df['pred_return_reg_10d'] > 0.3))
rets = df.loc[mask, 'true_return_reg_10d']
stats = strategy_stats(rets, 10)
print_strategy(stats, 10)
if stats: strategies['10d High-Conf'] = stats

# Strategy 3: Oversold Bounce
subheader('Strategy 3: "Oversold Bounce" (3-day hold)')
pr('  Rule: rsi_14 < 35 + bollinger_bands_signal == 1 + close < bbands_lower_20')
mask = ((df['rsi_14'] < 35) &
        (df['bollinger_bands_signal'] == 1) &
        (df['close'] < df['bbands_lower_20']))
rets = df.loc[mask, 'true_return_reg_3d']
stats = strategy_stats(rets, 3)
print_strategy(stats, 3)
if stats: strategies['Oversold Bounce'] = stats

# Strategy 3b: Relaxed Oversold + Model
subheader('Strategy 3b: "Oversold + Model Confirm" (3-day hold)')
pr('  Rule: rsi_14 < 40 + bull_bear_delta <= -2 + bin_3d=UP')
mask = ((df['rsi_14'] < 40) &
        (df['bull_bear_delta'] <= -2) &
        (df['pred_class_bin_3d'] == 1))
rets = df.loc[mask, 'true_return_reg_3d']
stats = strategy_stats(rets, 3)
print_strategy(stats, 3)
if stats: strategies['Oversold + Model'] = stats

# Strategy 4: 30-Day Trend Rider
subheader('Strategy 4: "30-Day Trend Rider" (30-day hold)')
pr('  Rule: bin_30d prob > 0.55 + reg_30d > 1.0 + sma_cross=1 + delta >= 1')
mask = ((df['prob_1_bin_30d'] > 0.55) &
        (df['pred_return_reg_30d'] > 1.0) &
        (df['sma_cross_signal'] == 1) &
        (df['bull_bear_delta'] >= 1))
rets = df.loc[mask, 'true_return_reg_30d']
stats = strategy_stats(rets, 30)
print_strategy(stats, 30)
if stats: strategies['30d Trend Rider'] = stats

# Strategy 5: Swing Catcher (bucket extremes)
subheader('Strategy 5: "Swing Catcher" (3-day hold, aggressive)')
pr('  Rule: buck_3d extreme buckets with concentrated probability')
# Bullish: upper buckets
bull_prob = df[['prob_5_buck_3d', 'prob_6_buck_3d', 'prob_7_buck_3d']].sum(axis=1)
bear_prob = df[['prob_0_buck_3d', 'prob_1_buck_3d', 'prob_2_buck_3d']].sum(axis=1)
bull_swing = (df['pred_class_buck_3d'] >= 5) & (bull_prob > 0.15)
bear_swing = (df['pred_class_buck_3d'] <= 2) & (bear_prob > 0.15)

bull_rets = df.loc[bull_swing, 'true_return_reg_3d']
bear_rets = df.loc[bear_swing, 'true_return_reg_3d']

pr(f'\n  Bullish swing trades: {len(bull_rets):,}')
if len(bull_rets) > 0:
    bstats = strategy_stats(bull_rets, 3)
    print_strategy(bstats, 3)
    strategies['Swing Bull'] = bstats

pr(f'\n  Bearish swing trades (short): {len(bear_rets):,}')
if len(bear_rets) > 0:
    short_rets = -bear_rets  # flip sign for short
    sstats = strategy_stats(short_rets, 3)
    print_strategy(sstats, 3)
    strategies['Swing Bear (short)'] = sstats

# Strategy 6: ML + Technical Confluence (10-day)
subheader('Strategy 6: "ML + Technical Confluence" (10-day hold)')
pr('  Rule: bin_10d=UP + reg_10d > 0.2 + macd_signal=1 + rsi 35-65 + adx > 20')
mask = ((df['pred_class_bin_10d'] == 1) &
        (df['pred_return_reg_10d'] > 0.2) &
        (df['macd_signal'] == 1) &
        (df['rsi_14'].between(35, 65)) &
        (df['adx_14'] > 20))
rets = df.loc[mask, 'true_return_reg_10d']
stats = strategy_stats(rets, 10)
print_strategy(stats, 10)
if stats: strategies['ML+Tech Confluence'] = stats

# Strategy 7: Conservative Monthly
subheader('Strategy 7: "Conservative Monthly" (30-day hold)')
pr('  Rule: bin_30d prob > 0.52 + reg_30d > 0 + beta 0.3-1.2 + PE 10-30 + delta >= 0')
mask = ((df['prob_1_bin_30d'] > 0.52) &
        (df['pred_return_reg_30d'] > 0) &
        (df['beta'].between(0.3, 1.2)) &
        (df['pe_ratio'].between(10, 30)) &
        (df['bull_bear_delta'] >= 0))
rets = df.loc[mask, 'true_return_reg_30d']
stats = strategy_stats(rets, 30)
print_strategy(stats, 30)
if stats: strategies['Conservative Monthly'] = stats

# Strategy 8: CCI Zero Cross + Model Confirm
subheader('Strategy 8: "CCI Zero Cross + Model Confirm" (10-day hold)')
pr('  Rule: cci_signal=1 + cci_14 in [0,50] + bin_10d=UP + macd_signal=1')
mask = ((df['cci_signal'] == 1) &
        (df['cci_14'].between(0, 50)) &
        (df['pred_class_bin_10d'] == 1) &
        (df['macd_signal'] == 1))
rets = df.loc[mask, 'true_return_reg_10d']
stats = strategy_stats(rets, 10)
print_strategy(stats, 10)
if stats: strategies['CCI Zero + Confirm'] = stats

# Strategy 9: Multi-Horizon Agreement
subheader('Strategy 9: "Multi-Horizon Agreement" (10-day hold)')
pr('  Rule: ALL 3 binary models agree UP + ALL 3 regression models > 0')
mask = ((df['pred_class_bin_3d'] == 1) &
        (df['pred_class_bin_10d'] == 1) &
        (df['pred_class_bin_30d'] == 1) &
        (df['pred_return_reg_3d'] > 0) &
        (df['pred_return_reg_10d'] > 0) &
        (df['pred_return_reg_30d'] > 0))
rets = df.loc[mask, 'true_return_reg_10d']
stats = strategy_stats(rets, 10)
print_strategy(stats, 10)
if stats: strategies['Multi-Horizon Agree'] = stats

# Strategy 10: Bucket Probability Spread
subheader('Strategy 10: "Probability Tilt" (10-day hold)')
pr('  Rule: buck_10d up-prob (buckets 4-7) > 0.55 + reg_10d > 0.5 + rsi < 70')
up_prob_10d = df[['prob_4_buck_10d', 'prob_5_buck_10d', 'prob_6_buck_10d', 'prob_7_buck_10d']].sum(axis=1)
mask = ((up_prob_10d > 0.55) &
        (df['pred_return_reg_10d'] > 0.5) &
        (df['rsi_14'] < 70))
rets = df.loc[mask, 'true_return_reg_10d']
stats = strategy_stats(rets, 10)
print_strategy(stats, 10)
if stats: strategies['Prob Tilt 10d'] = stats

# Strategy 11: Defensive Short (model says DOWN)
subheader('Strategy 11: "Defensive Short" (3-day hold)')
pr('  Rule: bin_3d=DOWN + reg_3d < -0.05 + rsi > 65 + bull_bear_delta <= -1')
mask = ((df['pred_class_bin_3d'] == 0) &
        (df['pred_return_reg_3d'] < -0.05) &
        (df['rsi_14'] > 65) &
        (df['bull_bear_delta'] <= -1))
rets = df.loc[mask, 'true_return_reg_3d']
short_rets = -rets  # flip for short profit
stats = strategy_stats(short_rets, 3)
print_strategy(stats, 3)
if stats: strategies['Defensive Short'] = stats


# ──────────────────────────────────────────────────────────────────────
# PART 4: TOP vs BOTTOM QUINTILE PROFILE
# ──────────────────────────────────────────────────────────────────────
header('PART 4: TOP vs BOTTOM QUINTILE DEEP DIVE')

for horizon, tag in [(3, 'reg_3d'), (10, 'reg_10d'), (30, 'reg_30d')]:
    pred_col = f'pred_return_{tag}'
    true_col = f'true_return_{tag}'
    if pred_col not in df.columns:
        continue
    q80, q20 = df[pred_col].quantile(0.8), df[pred_col].quantile(0.2)
    top = df[df[pred_col] >= q80]
    bot = df[df[pred_col] <= q20]

    subheader(f'{tag.upper()}: Top vs Bottom Quintile')
    pr(f'  {"Metric":<30s} {"Top 20%":<22s} {"Bottom 20%":<22s} {"Spread":<15s}')
    pr(f'  {"─"*30} {"─"*22} {"─"*22} {"─"*15}')

    for metric, col, fmt in [
        ('Avg Predicted Return', pred_col, '{:.4f}'),
        ('Avg True Return', true_col, '{:.4f}'),
        ('Win Rate', true_col, None),
        ('Avg RSI', 'rsi_14', '{:.1f}'),
        ('Avg Bull-Bear Delta', 'bull_bear_delta', '{:.1f}'),
        ('Avg Beta', 'beta', '{:.2f}'),
        ('Avg PE Ratio', 'pe_ratio', '{:.1f}'),
        ('Avg ADX', 'adx_14', '{:.1f}'),
        ('Avg CCI', 'cci_14', '{:.1f}'),
        ('Avg MACD', 'macd', '{:.3f}'),
        ('Avg 1d ROC', 'close_1d_roc', '{:.3f}'),
        ('Avg 52w High %', '52_week_high_pct', '{:.2f}'),
    ]:
        if col not in df.columns:
            continue
        if metric == 'Win Rate':
            top_val = f'{(top[col] > 0).mean():.1%}'
            bot_val = f'{(bot[col] > 0).mean():.1%}'
            spread = f'{(top[col]>0).mean() - (bot[col]>0).mean():.1%}'
        else:
            tv = top[col].mean()
            bv = bot[col].mean()
            top_val = fmt.format(tv)
            bot_val = fmt.format(bv)
            spread = fmt.format(tv - bv)
        pr(f'  {metric:<30s} {top_val:<22s} {bot_val:<22s} {spread:<15s}')


# ──────────────────────────────────────────────────────────────────────
# PART 5: v1 vs v2 COMPARISON
# ──────────────────────────────────────────────────────────────────────
header('PART 5: v1 vs v2 MODEL COMPARISON')

pr("""
  Architecture changes v1 -> v2:
  ┌──────────────────────────┬────────────────────────┬────────────────────────┐
  │ Component                │ v1                     │ v2                     │
  ├──────────────────────────┼────────────────────────┼────────────────────────┤
  │ Features                 │ 8 (price/vol only)     │ 46 (full technicals)   │
  │ Lookback                 │ 5 days                 │ 20 days                │
  │ Positional Encoding      │ None                   │ Sinusoidal + dropout   │
  │ Sequence Pooling         │ Last timestep only     │ Attention pooling      │
  │ Output Head              │ Single linear layer    │ 2-layer MLP + GELU     │
  │ Train/Val Split          │ Random shuffle         │ Temporal (by date)     │
  │ Normalization            │ Full dataset stats     │ Train-only stats       │
  │ Optimizer                │ Adam (no decay)        │ AdamW (weight decay)   │
  │ LR Schedule              │ Constant               │ Cosine + 3ep warmup    │
  │ Gradient Clipping        │ None                   │ max_norm=1.0           │
  │ Early Stopping           │ None (fixed epochs)    │ patience=7 + best ckpt │
  │ NaN Handling             │ Zero-fill              │ Forward-fill + zero    │
  └──────────────────────────┴────────────────────────┴────────────────────────┘

  v1 known issues (from analysis_model_improvements.txt):
    - reg_3d/reg_10d: correlation ~0.025, predicted ALL positive
    - bin_3d: always predicted "up" (recall=100%, precision=54.5%)
    - buck_3d: WORSE than random (7.6% vs 12.5%)
    - Only bin_30d and reg_30d showed meaningful discrimination
""")


# ──────────────────────────────────────────────────────────────────────
# PART 6: STRATEGY COMPARISON DASHBOARD
# ──────────────────────────────────────────────────────────────────────
header('PART 6: STRATEGY COMPARISON DASHBOARD')

pr(f'\n  {"Strategy":<28s} {"Trades":<8s} {"Win %":<8s} {"Avg Ret":<10s} {"Med Ret":<10s} {"Sharpe":<8s} {"Grade":<8s}')
pr(f'  {"═"*28} {"═"*8} {"═"*8} {"═"*10} {"═"*10} {"═"*8} {"═"*8}')

for name, stats in sorted(strategies.items(), key=lambda x: -x[1].get('sharpe', 0)):
    trades = stats['trades']
    wr = stats['win_rate']
    avg = stats['avg_ret']
    med = stats['median_ret']
    sharpe = stats['sharpe']

    if sharpe > 1.5 and wr > 0.60:
        grade = 'A'
    elif sharpe > 1.0 and wr > 0.55:
        grade = 'A-'
    elif sharpe > 0.8 and wr > 0.55:
        grade = 'B+'
    elif sharpe > 0.5 and wr > 0.52:
        grade = 'B'
    elif avg > 0 and wr > 0.50:
        grade = 'C'
    elif avg > 0:
        grade = 'C-'
    else:
        grade = 'F'

    pr(f'  {name:<28s} {trades:<8d} {wr:<8.1%} {avg:<+10.3f} {med:<+10.3f} {sharpe:<8.2f} {grade:<8s}')


# ──────────────────────────────────────────────────────────────────────
# PART 7: RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────
header('PART 7: RECOMMENDATIONS')

# Sort strategies by grade/sharpe
ranked = sorted(strategies.items(), key=lambda x: -x[1].get('sharpe', 0))
stable = [(n, s) for n, s in ranked if s['sharpe'] > 0.5 and s['win_rate'] > 0.52]
aggressive = [(n, s) for n, s in ranked if s['sharpe'] > 0 and (n, s) not in stable]

pr('\n  STABLE STRATEGIES (add to daily rotation):')
for i, (name, s) in enumerate(stable[:5], 1):
    pr(f'    {i}. {name} — {s["win_rate"]:.0%} win, {s["avg_ret"]:+.2f}% avg, Sharpe {s["sharpe"]:.2f}')

pr('\n  AGGRESSIVE STRATEGIES (smaller allocation):')
for i, (name, s) in enumerate(aggressive[:5], 1):
    pr(f'    {i}. {name} — {s["win_rate"]:.0%} win, {s["avg_ret"]:+.2f}% avg, Sharpe {s["sharpe"]:.2f}')

pr("""
  KEY TAKEAWAYS:
    - v2 models now see ALL 46 technical features (was 8)
    - Temporal split eliminates look-ahead bias (honest val metrics)
    - Attention pooling uses full 20-day sequence (was last-step only)
    - Early stopping prevents overfitting (all models stopped before 30 epochs)
    - All 9 models trained successfully (was only 7 in v1)

  NEXT STEPS:
    - Run extended backtest on full CSV data (2016-2025) for OOS validation
    - Consider ensemble weighting: inverse-volatility across strategies
    - Implement top strategies as live runners in strategies/ directory
    - Retrain with extended data window if more CSV data becomes available
""")

header('END OF REPORT')

# Write to file
with open('analysis_strategy_proposals_v2.txt', 'w') as f:
    f.write('\n'.join(lines))

print(f'\n\nSaved to: analysis_strategy_proposals_v2.txt')
