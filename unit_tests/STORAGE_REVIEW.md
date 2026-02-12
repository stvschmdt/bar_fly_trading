# storage.py Review — Issues & Recommendations

Reviewed: 2026-02-11
File: `api_data/storage.py`
Test file: `unit_tests/test_storage.py` (42 tests, all passing)

---

## Critical Bugs

### 1. Stock split adjustment off-by-one
**Location:** `adjust_for_stock_splits()` — [storage.py:211](../api_data/storage.py#L211)
**Current:** `mask = (df['date'] <= effective_date)`
**Should be:** `mask = (df['date'] < effective_date)`

The `effective_date` is the first day of trading at the NEW (post-split) price. Using `<=` incorrectly divides the already-post-split price on that day. Example: NVDA 10:1 split on 2024-06-10 — open was ~$120 (post-split). The `<=` mask divides it by 10, producing ~$12.

**Fix:** Change `<=` to `<` on line 211.
**Test:** `TestSplitAdjustment::test_split_off_by_one_current_behavior` proves the bug.

---

### 2. `inflation` missing from ffill block
**Location:** [storage.py:359-366](../api_data/storage.py#L359-L366)
**Issue:** The SQL query selects `econ.inflation`, but the ffill block forward-fills 8 of 9 economic columns — `inflation` is skipped. Since inflation data is annual, most trading days have NaN.

**Fix:** Add `df['inflation'] = df['inflation'].ffill()` after line 366.
**Test:** `TestFfill::test_inflation_not_ffilled_bug` confirms the gap.

---

### 3. SQL injection via f-string concatenation
**Locations:**
- `insert_ignore_data()` line 106 — string values via `f"'{row[col]}'"`
- `select_all_by_symbol()` line 116 — symbols via f-string
- `get_dates_for_symbol()` line 153 — symbol and dates via f-string

**Risk:** Low with controlled ticker inputs, but a value containing a single quote (e.g., in options contract data) would break or inject SQL. Also inconsistent with `get_last_updated_date()` which correctly uses `:symbol` parameter binding.

**Fix:** Use parameterized queries (`:param` binding) everywhere.

---

## ML / Data Quality Concerns

### 4. Look-ahead bias from company_overview JOIN
**Location:** [storage.py:317](../api_data/storage.py#L317)
**Issue:** `company_overview` is a single current-snapshot row per symbol. The JOIN (`ON core.symbol = comp.symbol`, no date filter) puts TODAY's values on every historical row:
- `52_week_high` / `52_week_low` — current range, not what it was in 2020
- `book_value` — current book value used for all historical `price_to_book_ratio` (line 356)
- `beta`, `market_cap`, `forward_pe`, `shares_outstanding` — all current
- `analyst_rating_*` — today's consensus applied to 2018 rows

The ffill on lines 352-353 (`52_week_high/low`) is a no-op — the JOIN already fills every row with the same value.

**Impact:** Training data leaks future information into features.
**Possible fixes:**
- Store periodic company_overview snapshots with dates
- Exclude snapshot-only fields from ML features (they're already excluded from `BASE_FEATURE_COLUMNS` in config.py — but `52_week_high_pct` and `52_week_low_pct` ARE used)
- Compute rolling 52-week high/low from historical price data instead

---

### 5. Division-by-zero in derived ratios
**Locations:**
- `pcr` (line 337): `put_volume / call_volume` → `inf` when `call_volume = 0`
- `pe_ratio` (line 355): `adjusted_close / ttm_eps` → `inf` when `ttm_eps = 0`
- `price_to_book_ratio` (line 356): `adjusted_close / book_value` → `inf` when `book_value = 0`

The `inf` values propagate into rolling means (`pcr_14_mean`) and downstream ML features.

**Fix:** Clip inf to NaN or a reasonable max:
```python
df['pcr'] = (df['put_volume'] / df['call_volume']).clip(upper=10.0).round(2)
df['pe_ratio'] = (df['adjusted_close'] / df['ttm_eps']).replace([np.inf, -np.inf], np.nan)
```
**Tests:** `TestOptionsDerived::test_pcr_division_by_zero`, `TestDerivedRatios::test_pe_ratio_zero_eps`

---

### 6. NaN handling in insert_ignore_data
**Location:** [storage.py:106](../api_data/storage.py#L106)
**Issue:** `str(row[col])` for float columns produces the literal string `nan` when value is NaN. This is invalid SQL — should be `NULL`.

**Fix:** Check for NaN before converting:
```python
val = row[col]
if pd.isna(val):
    return 'NULL'
elif col_type not in {int, float}:
    return f"'{val}'"
else:
    return str(val)
```
**Test:** `TestInsertIgnoreData::test_nan_float_produces_invalid_sql`

---

## Performance Issues

### 7. O(n^2) signal generation loop
**Location:** [storage.py:419-434](../api_data/storage.py#L419-L434)
**Issue:** The loop does `df[(df.symbol == row['symbol']) & (df.date == row['date'])]` on every row. For 5M rows in a batch, this is O(n * n_total) boolean comparisons — catastrophically slow.

**Also:** `StockScreener` is initialized with `df.head()` (line 411) — only 5 rows. If any `_check` function references `self.df` for lookups, it would get incomplete data.

**Fix options:**
1. Vectorize the `_check` functions to operate on full columns (preferred)
2. Group by symbol first, then iterate within group
3. At minimum, use `df.itertuples()` with iloc instead of boolean mask filtering

---

### 8. store_data index column artifact
**Location:** [storage.py:91](../api_data/storage.py#L91)
**Issue:** `df.to_sql(..., index=include_index)` with `include_index=True` (default). If the DataFrame has a default RangeIndex, this writes an unwanted `index` column. This is likely the source of the `Unnamed: 0` column in `all_data_*.csv` files.

**Fix:** Default `include_index=False` for tables that don't need it, or set meaningful index before calling.

---

## Suggested Derived Fields for ML

These fields could be added in `gold_table_processing()` after the existing derived columns. Tests for all three exist in `TestSuggestedDerivedFields`.

| Field | Formula | Rationale |
|-------|---------|-----------|
| `overnight_gap` | `(open - prev_close) / prev_close` | Pre-market sentiment; captures gap-up/gap-down dynamics that intraday features miss |
| `intraday_range` | `(high - low) / open` | Daily volatility measure without close-to-close noise; useful for regime detection |
| `garman_klass_vol` | `0.5 * ln(H/L)^2 - (2*ln2-1) * ln(C/O)^2` | More efficient volatility estimator from OHLC — theoretically 5-8x more efficient than close-to-close |

### Note on existing feature pipeline
The following features are already computed downstream (NOT in storage.py):
- `close_1d_roc`, `close_3d_roc`, `close_10d_roc`, `close_5d_vol`, `vol_3d_mean`, `vol_10d_mean` — in `stockformer/features.py`
- `sector_etf_ret_*`, `spy_ret_*`, `sector_rel_ret_1d` — in `stockformer/sector_features.py`
- `adjusted_close` is renamed to `close` at load time in `stockformer/data_utils.py`

Storage.py does NOT need to duplicate those.

---

## Priority Order for Fixes
1. **Split off-by-one** (#1) — corrupts historical adjusted prices
2. **Inflation ffill** (#2) — one-line fix, immediate data quality improvement
3. **Division-by-zero** (#5) — prevents inf propagation into features
4. **Look-ahead bias** (#4) — design decision needed, biggest ML impact
5. **SQL injection** (#3) — low urgency with controlled inputs
6. **Signal loop performance** (#7) — matters if re-running gold table processing
7. **Derived fields** — nice to have, improves feature set
