# Scaling Plan: 600 → 1,000 Symbols

**Date:** 2026-02-13
**Status:** Draft — sizing analysis

---

## 1. Current Baseline (600 symbols)

### EC2 Instance (data pull, web serving, IBKR gateway)
| Resource | Current | Capacity | Utilization |
|----------|---------|----------|-------------|
| Instance type | t3 (2 vCPU Xeon 8259CL, 3.7 GB RAM) | — | — |
| Disk | 113 GB used | 128 GB | **88%** |
| MySQL | **67 GB** | — | Dominates disk |
| RAM | 1.9 GB used + 883 MB swap | 3.7 GB | **Swap active** |
| CSVs (all_data_*) | 987 MB (37 files, ~27 MB each) | — | — |
| Symbols | 600 (502 S&P 500 + 86 watchlist + 12 ETFs) | — | — |

### Nightly Pipeline Timing (2026-02-13 run)
| Step | Duration | Notes |
|------|----------|-------|
| 1. Data pull + gold tables | **1h 18m** | Alpha Vantage, 140 req/min |
| 2. Screener PDF + Drive | 2 min | 110-page PDF |
| 3. SCP to DGX | ~1 min | 987 MB over Tailscale |
| 4. DGX inference (9 models) | ~2 min | 3 parallel groups |
| 5. SCP from DGX | <1 min | ~39 MB merged CSV |
| 6. Web data refresh | 3 min | 544 JSONs + 544 history files |
| **Total** | **~1h 27m** | Steps 3-5 failed tonight (DGX busy training) |

### Training (DGX GB10 GPU)
| Metric | Value |
|--------|-------|
| Training rows | 1,290,934 (532 tickers x ~2,430 days) |
| Features | 41 stock + 11 market |
| Model params | 1.6M (cross-attention) |
| GPU memory | ~2 GB per model, 3 in parallel |
| Epoch time | ~10-15 min per model |
| Full 9-model run | est. 5-8 hours |

---

## 2. Projected at 1,000 Symbols (+67%)

### Where the 400 new symbols come from
Options (not mutually exclusive):
- **Russell 1000 complement**: ~400 mid-caps not in S&P 500
- **Sector fill**: under-represented sectors (materials, utilities, energy)
- **High-volume small caps**: liquid names with options data
- **International ADRs**: TSM, BABA, NVO, etc.

### Resource Scaling Estimates

| Component | Current (600) | Projected (1,000) | Growth | Bottleneck? |
|-----------|--------------|-------------------|--------|-------------|
| **API data pull** | 1h 18m | **~25-35 min** | 0.4x | Premium 600 req/min |
| **MySQL disk** | 67 GB | **~112 GB** | 1.67x | **EXCEEDS 128 GB disk** |
| **CSV files** | 987 MB (37 files) | ~1.65 GB (~62 files) | 1.67x | Moderate |
| **Training rows** | 1.29M | ~2.15M | 1.67x | Longer epochs |
| **Training epoch** | 10-15 min | ~17-25 min | 1.67x | Linear with data |
| **GPU memory** | ~2 GB/model | ~2.2 GB/model | ~1.1x | Batch size stays 256 |
| **Inference** | ~2 min (9 models) | ~3-4 min | ~1.8x | I/O bound |
| **Web JSON files** | 1,088 (544x2) | 2,000 (1000x2) | 1.84x | Disk + gen time |
| **Web refresh** | 3 min | ~5 min | 1.67x | CSV read, linear |
| **SCP to DGX** | ~1 min | ~2 min | 1.65x | Network, linear |
| **Merged predictions** | 39 MB | ~65 MB | 1.67x | Negligible |
| **Total nightly** | 1h 27m | **~2h 25m** | 1.66x | Within overnight window |

---

## 3. Blockers

### 3.1 Disk Space (CRITICAL)
The EC2 has **15 GB free** on a 128 GB volume. MySQL is 67 GB and would grow to ~112 GB with 1,000 symbols. This alone exceeds capacity before accounting for CSVs, logs, and web data.

**Options:**
1. **Expand EBS volume** to 256 GB ($0.08/GB/mo = ~$10/mo incremental). Non-disruptive resize while running.
2. **Prune historical_options table** — likely the largest table by far. If we only need recent options data (last 2 years), significant space reclaimed.
3. **Move MySQL to RDS** — offload DB entirely. Adds latency to data pull but eliminates disk pressure. Cost: ~$15-30/mo for db.t3.medium.
4. **Compress old partitions** — InnoDB table compression can save 30-50%.

**Recommendation:** Expand EBS to 256 GB first (low-risk, immediate). Then audit `historical_options` table — it's likely 50+ GB on its own and most of that data is rarely queried.

### 3.2 RAM (WARNING)
With 3.7 GB total and active swap (883 MB), the EC2 is already memory-constrained. The gold table build, MySQL InnoDB buffer pool, web server, and IB Gateway all compete for RAM. Adding 67% more symbols increases working set size.

**Options:**
1. **Upgrade to t3.large** (8 GB RAM, 2 vCPU) or **t3.xlarge** (16 GB, 4 vCPU).
2. **Increase swap** to 4 GB (free, but slower).
3. **Tune MySQL** — reduce `innodb_buffer_pool_size` to leave more for Python.

**Recommendation:** Upgrade to t3.large (8 GB RAM, same 2 vCPU). The extra 4 GB eliminates swap usage and gives MySQL a proper buffer pool.

### 3.3 Alpha Vantage Rate Limit (NON-ISSUE)
We have Alpha Vantage Premium, which at 1,000 symbols would give us 600 req/min:
- 1,000 symbols x ~5 calls each = ~5,000 calls ÷ 600/min → **~8 min raw, ~25-35 min with overhead**

This is a massive speedup from the current 1h 18m (which runs at 140 req/min free-tier pacing).

---

## 4. Timing Budgets

### 4.1 Batch Window: 4:30 PM → 6:00 AM ET (13.5 hours)

All non-real-time jobs must complete in this window: data pull, gold tables, screener, SCP, inference, web refresh, and (when scheduled) model retraining.

| Job | Current (600) | Projected (1,000) | Window Budget |
|-----|--------------|-------------------|---------------|
| Data pull + gold tables | 1h 18m | ~25-35m (AV premium 600/min) | — |
| Screener PDF | 2m | ~3m | — |
| SCP + inference + SCP | 4m | ~7m | — |
| Web refresh | 3m | ~5m | — |
| **Nightly subtotal** | **1h 27m** | **~45-55m** | **13.5h available** |
| Model retraining (periodic) | 5-8h | 8-14h | Same window or weekend |

At 1,000 symbols with AV premium, the nightly pipeline actually gets **faster** than today (~45 min vs 1h 27m). Data pull drops from 1h18m to ~30m thanks to 600 req/min. Model retraining at 8-14 hours fits in a single overnight; could also run on weekends.

### 4.2 Real-Time Window: 15 min target, 30 min max

During market hours, `rt_scan_loop.sh` runs on a cycle. Each cycle must:
1. Scan all symbols for strategy signals
2. Execute pending orders via IBKR
3. Update position tracking

| Component | Current (600) | Projected (1,000) | Constraint |
|-----------|--------------|-------------------|------------|
| Strategy signal scan | ~3-5 min | ~5-8 min | Linear with symbols |
| IBKR order execution | <1 min | <1 min | Depends on signal count, not universe |
| Position tracking | <1 min | <1 min | Only open positions |
| **Cycle total** | **~5-7 min** | **~7-10 min** | **15 min target OK** |

At 1,000 symbols the RT scan cycle fits within the 15-min target. If it approaches 12-13 min, options to speed up:
- **Parallelize strategy scans** across symbols (currently sequential within each strategy)
- **Pre-filter**: only scan symbols with fresh signals in merged_predictions (skip low-probability names)
- **Reduce scan to top-N**: only RT-scan the top 200-300 ranked symbols per strategy

**Key risk:** If the scan + execution exceeds 15 min, we miss the next scan window and signals go stale. At 1,000 symbols this is unlikely but should be monitored.

---

## 5. DGX / Training Impact

### Training time
- Current: ~10-15 min/epoch with 1.29M rows at batch_size=256
- Projected: ~17-25 min/epoch with 2.15M rows
- Full 9-model training (80 epoch max, ~15-25 effective with early stopping): **8-14 hours** (up from 5-8)
- Fits in overnight batch window; retrain weekly or on weekends for comfort

### GPU memory
- Batch_size stays 256 — more rows just means more batches per epoch
- Memory increase is negligible (<10%)
- GB10 has plenty of headroom — no upgrade needed

### Model quality considerations
- More symbols = more diverse training signal = better generalization
- Mid-cap stocks are noisier — may want to add `market_cap` as a feature or weight samples by liquidity
- Consider: universe shift could change optimal hyperparams (especially noisy-label threshold)

---

## 5. Stretch Scenario: 5,000 Symbols (8.3x current)

This covers the full Russell 3000 + ~2,000 additional liquid names (large-cap international ADRs, high-volume OTC, popular ETFs). This is a fundamentally different operating regime that stresses every component.

### 5.1 Where 5,000 symbols come from
- Russell 3000: ~3,000 US equities (large + mid + small cap)
- International ADRs: ~500 (TSM, BABA, NVO, SAP, ASML, etc.)
- Liquid ETFs: ~200 (sector, thematic, leveraged, fixed income)
- High-volume OTC / recently-IPO'd: ~300
- Remaining watchlist / custom: ~1,000

### 5.2 Resource Scaling Estimates

| Component | Current (600) | 1,000 | 5,000 | Growth (5K) |
|-----------|--------------|-------|-------|-------------|
| **API data pull** | 1h 18m | ~25-35m | **~42 min** (AV premium) | 0.5x |
| **MySQL disk** | 67 GB | ~112 GB | **~560 GB** | 8.3x |
| **CSV files** | 987 MB (37 files) | ~1.65 GB | **~8.2 GB (~310 files)** | 8.3x |
| **Training rows** | 1.29M | ~2.15M | **~10.8M** | 8.3x |
| **Training epoch** | 10-15 min | ~17-25 min | **~80-120 min** | 8.3x |
| **GPU memory** | ~2 GB/model | ~2.2 GB | **~3-4 GB/model** | ~1.5-2x |
| **Inference** | ~2 min | ~3-4 min | **~15-20 min** | 8-10x |
| **Web JSON files** | 1,088 | 2,000 | **200 pre-gen + on-demand** | Capped |
| **Web refresh** | 3 min | ~5 min | **~5 min** (top 100 only) | Flat |
| **SCP to DGX** | ~1 min | ~2 min | **~9 min** | 8.2x |
| **Merged predictions** | 39 MB | ~65 MB | **~325 MB** | 8.3x |
| **Total nightly** | 1h 27m | ~45-55m | **~1h 30m** | ~1x |
| **RT scan cycle** | ~5-7 min | ~7-10 min | **~40-60 min** | 8x |

### 5.3 What Breaks

**API data pull (MANAGEABLE with AV Premium):**
We have AV Premium (600 req/min). At 5,000 symbols x ~5 calls = 25,000 calls ÷ 600/min → **~42 min raw, ~1 hour with retries**. This is well within the batch window and actually faster than today's 600-symbol run on the free tier.

If AV Premium ever becomes a bottleneck, bulk-data providers are an option:

| API Provider | Price | Approach | Pull Time (5K symbols) |
|-------------|-------|----------|----------------------|
| AV Premium (current) | $50/mo | Per-symbol, 600/min | **~1 hour** |
| Polygon.io Starter | $29/mo | Bulk snapshot | ~30 min |
| Polygon.io Developer | $79/mo | Full websocket + bulk | ~10 min |
| Tiingo | $30/mo | Bulk daily download | ~15 min |

**Verdict at 5K:** AV Premium handles it. No API change needed.

**Disk (BREAKS):**
MySQL at 560 GB needs a fundamentally different storage strategy.

| Option | Size | Cost |
|--------|------|------|
| EBS gp3 1 TB | 1 TB | ~$80/mo |
| EBS gp3 2 TB | 2 TB | ~$160/mo |
| RDS db.t3.large (1 TB) | 1 TB + managed | ~$150/mo |
| S3 + Parquet (cold storage) | unlimited | ~$5-10/mo |

**Recommendation at 5K:** EBS 1 TB ($80/mo) + prune historical_options aggressively. Or migrate historical data to S3 Parquet and keep MySQL for hot data only (last 90 days).

**RAM (BREAKS):**
Loading 8.2 GB of CSVs into memory for gold table generation or training data prep will OOM on anything less than 16 GB. The training DataLoader handles this via batching, but the initial CSV read (`pd.read_csv` for 310 files) needs careful streaming.

| Instance | RAM | vCPU | Price |
|----------|-----|------|-------|
| t3.large | 8 GB | 2 | $60/mo |
| t3.xlarge | 16 GB | 4 | $120/mo |
| m5.xlarge | 16 GB | 4 | $140/mo |
| m5.2xlarge | 32 GB | 8 | $280/mo |
| r5.xlarge | 32 GB | 4 | $180/mo (memory-optimized) |

**Recommendation at 5K:** m5.xlarge (16 GB, 4 vCPU) minimum. r5.xlarge (32 GB, 4 vCPU) if MySQL stays local.

**Real-time scan (BREAKS):**
At 40-60 min per cycle, the RT scan far exceeds the 30 min max. This requires an architectural change:

1. **Pre-filter to top-N**: Only RT-scan the top 500-1000 symbols ranked by model confidence. Reduces effective universe to ~1,000 for real-time.
2. **Parallel scanning**: Split universe across 4-8 worker processes.
3. **Incremental scan**: Only re-scan symbols whose price moved >0.5% since last scan.
4. **Tiered universe**: Tier 1 (top 200) scanned every 5 min, Tier 2 (next 800) every 15 min, Tier 3 (rest) every 30 min.

**Recommendation at 5K:** Pre-filter + tiered scanning. No need to RT-scan 5,000 symbols every cycle.

**Training (STRETCHED):**
10.8M rows at 80-120 min/epoch. With 80 epochs max and early stopping at ~20, one model takes ~26-40 hours. Three parallel models = ~26-40 hours total on a single GPU. The full 9-model run: **3-5 days**.

| Option | Time for 9 models | Cost |
|--------|-------------------|------|
| GB10 (current, 1 GPU) | 3-5 days | $0 |
| Multi-GPU (2x GB10 or A100) | 1.5-2.5 days | Hardware dependent |
| Cloud GPU (A100 spot) | 1-2 days | ~$1-3/hr = $24-72 |
| Reduce epochs + aggressive early stopping | 2-3 days | $0 |
| Subsample training data (e.g., 50%) | 1.5-2.5 days | $0 |

**Recommendation at 5K:** Accept multi-day training (run over weekends), or subsample training data to ~5M rows. The model likely doesn't benefit linearly from 10M+ rows due to diminishing returns and noise in small-cap data.

**DGX inference (STRETCHED but OK):**
15-20 min for 9 models is within budget for nightly batch. For real-time, inference is only run on the pre-filtered universe (~500-1000 symbols), which stays at ~2-4 min.

### 5.4 Cost Summary: 5,000 Symbols

| Item | Current | 1,000 Symbols | 5,000 Symbols |
|------|---------|---------------|---------------|
| EC2 instance | ~$15 (t3.small) | ~$60 (t3.large) | **~$140 (m5.xlarge)** |
| EBS storage | ~$10 (128 GB) | ~$20 (256 GB) | **~$80 (1 TB)** |
| Data API (AV Premium) | $50 (existing) | $50 | $50 |
| DGX (own) | $0 | $0 | $0 |
| Cloud GPU (optional) | $0 | $0 | ~$50-100 (spot, periodic) |
| **Total** | **~$25/mo** | **~$80-130/mo** | **~$250-400/mo** |

### 5.5 Architecture Changes Required at 5K

| Change | Why | Effort |
|--------|-----|--------|
| Bump AV Premium rate limit | Ensure 600 req/min is enabled | Low — config change |
| Streaming CSV reads | 8 GB CSVs won't fit in RAM | Low — pandas chunked reads |
| Tiered RT scanning | 40-60 min cycle exceeds 30 min max | Medium — new scheduler |
| MySQL partitioning / S3 offload | 560 GB DB needs archival strategy | Medium — migration script |
| Training data subsampling | 10.8M rows = multi-day training | Low — add --sample-frac flag |
| Web: top-100 pre-gen + on-demand | No need to pre-gen 10K JSONs | Medium — add API route for on-demand symbol lookup |

---

## 6. Implementation Plan

### Phase 1: EC2 Infrastructure (do first)
- [ ] Expand EBS volume from 128 GB to 256 GB
- [ ] Upgrade instance to t3.large (8 GB RAM)
- [ ] Verify MySQL restarts cleanly after resize
- [ ] Tune `innodb_buffer_pool_size` for new RAM (3-4 GB)
- [ ] Audit `historical_options` table size — prune if >40 GB

### Phase 2: Symbol Universe Expansion
- [ ] Curate 400 new symbols (Russell 1000 complement + high-liquidity names)
- [ ] Create `api_data/extended_universe.csv` or expand `sp500.csv`
- [ ] Run initial historical backfill (one-time full pull, ~4-6 hours)
- [ ] Verify gold table generation handles new symbols
- [ ] Check for symbols with missing fundamental data (smaller companies may lack analyst coverage)

### Phase 3: Pipeline Validation
- [ ] Run full nightly pipeline with 1,000 symbols end-to-end
- [ ] Verify CSV file count and sizes
- [ ] Confirm training and inference work on expanded dataset
- [ ] Monitor EC2 disk/RAM/swap during full run
- [ ] Test web frontend with 1,000 symbols (big board, sector map, stock detail)

### Phase 4: Model Retraining
- [ ] Retrain all 9 models on expanded universe
- [ ] Compare prediction quality vs. 600-symbol baseline
- [ ] Evaluate if mid-cap additions improve or dilute signal
- [ ] Consider market_cap weighting or liquidity-based sample weights

---

## 7. Cost Comparison (All Three Tiers)

| Item | 600 (today) | 1,000 | 5,000 |
|------|-------------|-------|-------|
| EC2 instance | $15 (t3.small) | $60 (t3.large) | $140 (m5.xlarge) |
| EBS storage | $10 (128 GB) | $20 (256 GB) | $80 (1 TB) |
| Data API (AV Premium) | $50 | $50 | $50 |
| Cloud GPU (optional) | — | — | $50-100 (spot) |
| DGX (own) | $0 | $0 | $0 |
| **Total** | **~$25/mo** | **~$80-130/mo** | **~$250-400/mo** |

---

## 8. Architecture Decision: CSVs vs Direct DB Reads

At 5,000 symbols the CSV pipeline (MySQL → all_data_*.csv → SCP to DGX → training/inference) starts to feel heavy: 8.2 GB of files, 310 CSVs, and a multi-minute SCP. Since everything lives on the EC2 MySQL server, the question is: **should we skip CSVs entirely and read from the DB?**

### Current flow (CSV-based)
```
EC2 MySQL → gold_table.py → all_data_*.csv (disk) → SCP → DGX → training/inference
```

### Alternative: Direct DB reads
```
EC2 MySQL → (network) → DGX training/inference reads DB directly via SQLAlchemy
```

### Trade-offs

| Factor | CSV (current) | Direct DB |
|--------|--------------|-----------|
| **DGX training** | Reads local files (fast I/O) | Network reads to EC2 MySQL (latency) |
| **Data freshness** | Snapshot at gold table time | Always current |
| **Disk on EC2** | 987 MB → 8.2 GB of CSVs | No CSV storage needed |
| **Disk on DGX** | Same CSV copy | No CSV storage needed |
| **SCP step** | 1-9 min depending on size | Eliminated |
| **Reproducibility** | CSVs are a frozen snapshot | DB can change between runs |
| **Failure mode** | If DB is down, CSVs still work | DB down = no training |
| **Complexity** | Gold table script + SCP | DB connection from DGX, auth, firewall |

### Recommendation: Hybrid approach at 5K

1. **Keep CSVs for training** — DGX reads local files at full NVMe speed. Reading 8 GB over Tailscale from EC2 MySQL would add latency to every epoch. Training happens infrequently (weekly), so the SCP cost is acceptable.

2. **Drop CSVs for inference** — Inference reads ~1 day of data for 5K symbols (~5K rows). This is a tiny query that MySQL can serve in seconds. The DGX could query EC2 MySQL directly, eliminating the SCP-to/SCP-from dance for nightly inference.

3. **Drop CSVs for web refresh** — `populate_all.py` and `generate_history.py` already run on EC2 where MySQL lives. They could read from DB directly instead of parsing CSVs. At 5K symbols, parsing 310 CSVs to extract the latest row per symbol is wasteful when a single `SELECT ... GROUP BY symbol ORDER BY date DESC LIMIT 1` per symbol would suffice.

4. **Keep gold table generation** — Still needed to materialize the joined/feature-engineered view. But output could go to a MySQL materialized table instead of CSVs, serving as the single source of truth.

### What this buys us
- **Eliminates ~8 GB of CSV disk** on both EC2 and DGX
- **Eliminates SCP steps 3 and 5** from nightly pipeline (~10 min saved at 5K)
- **Web refresh reads DB** instead of parsing 310 CSVs (faster, less RAM)
- **Training still gets fast local reads** via periodic SCP of the gold table

---

## 9. Open Questions

1. **Which 400 symbols?** Russell 1000 complement, or hand-curated high-volume names?
2. **Historical backfill depth?** Full history (2016+) vs. recent only (2020+)? Full gives more training data but takes longer.
3. **Options data scope?** `historical_options` is likely the MySQL size driver. Do we need options for all 1,000, or just the top 200 by volume?
4. **Alpha Vantage premium?** $50/mo halves pull time. Worth it at 1,000 symbols?
5. **Webapp pagination?** Big board shows all 544 symbols. At 1,000, need server-side filtering/pagination?
6. **Stale symbol cleanup?** Some current symbols may be delisted or illiquid. Prune before expanding?
