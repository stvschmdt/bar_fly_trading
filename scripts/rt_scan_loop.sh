#!/bin/bash
# Real-time multi-strategy scan loop
#
# Runs ALL strategy runners in live mode on a repeating interval during
# market hours (9:30am - 4:00pm ET, Mon-Fri).  Each scan:
#   1. Fetches bulk realtime quotes from AlphaVantage (1 call per 100 symbols)
#   2. Merges with overnight data (technicals + ML predictions)
#   3. Runs all 5 strategies against the merged data
#   4. Aggregates signals into a single pending_orders.csv
#
# Usage:
#   ./scripts/rt_scan_loop.sh                          # default: 15 min interval
#   ./scripts/rt_scan_loop.sh --interval 5             # every 5 minutes
#   ./scripts/rt_scan_loop.sh --strategies "bollinger oversold_bounce"
#   ./scripts/rt_scan_loop.sh --dry-run                # print what would run
#   ./scripts/rt_scan_loop.sh --once                   # run once and exit
#
# Output:
#   signals/pending_orders.csv  — aggregated signals from all strategies
#   logs/rt_scan.log            — append-only log of all scan runs
#
# Requires:
#   - Alpha Vantage API key (ALPHAVANTAGE_API_KEY env var)
#   - merged_predictions.csv (for ML strategies)
#   - all_data_*.csv in DATA_DIR (for technical strategies)

set -e

cd ~/proj/bar_fly_trading

# ── Config ──────────────────────────────────────────────────────────
INTERVAL_MIN=15
SIGNAL_DIR="signals"
SIGNAL_FILE="$SIGNAL_DIR/pending_orders.csv"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/rt_scan.log"
DRY_RUN=false
ONCE=false
EXECUTE=false          # --execute: send signals to IBKR immediately after scan
LIVE_FLAG=""           # --live: real money trading
CLIENT_ID=10
GATEWAY_FLAG="--gateway"
DEFAULT_SHARES=""      # --default-shares N: fixed share count (0 or empty = auto-size)
NO_NOTIFY=""           # --no-notify: suppress email notifications
MARKET_ORDERS=""       # --market-orders: use market orders instead of limit
BUY_ONLY=""            # --buy-only: skip SELL signals for symbols we don't hold

# Data sources
PREDICTIONS="${PREDICTIONS:-merged_predictions.csv}"
DATA_DIR="${DATA_DIR:-/home/stvschmdt/data}"
DATA_PATH="$DATA_DIR/all_data_*.csv"

# Default: all 5 strategies
ALL_STRATEGIES="bollinger oversold_bounce oversold_reversal low_bounce regression_momentum"
STRATEGIES="$ALL_STRATEGIES"

# Market hours in ET (24h format)
MARKET_OPEN_H=9
MARKET_OPEN_M=30
MARKET_CLOSE_H=16
MARKET_CLOSE_M=0

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)      INTERVAL_MIN="$2"; shift 2 ;;
        --strategies)    STRATEGIES="$2"; shift 2 ;;
        --predictions)   PREDICTIONS="$2"; shift 2 ;;
        --data-dir)      DATA_DIR="$2"; DATA_PATH="$DATA_DIR/all_data_*.csv"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --once)          ONCE=true; shift ;;
        --execute)       EXECUTE=true; shift ;;
        --default-shares) DEFAULT_SHARES="--default-shares $2"; shift 2 ;;
        --no-notify)     NO_NOTIFY="--no-notify"; shift ;;
        --market-orders) MARKET_ORDERS="--market-orders"; shift ;;
        --buy-only)      BUY_ONLY="--buy-only"; shift ;;
        --live)          LIVE_FLAG="--live"; shift ;;
        --client-id)     CLIENT_ID="$2"; shift 2 ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

INTERVAL_SEC=$((INTERVAL_MIN * 60))

# ── Setup ───────────────────────────────────────────────────────────
mkdir -p "$SIGNAL_DIR" "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

is_market_hours() {
    local now_et
    now_et=$(TZ="America/New_York" date '+%H%M')
    local day_of_week
    day_of_week=$(TZ="America/New_York" date '+%u')  # 1=Mon, 7=Sun

    # Weekend check
    if [[ $day_of_week -ge 6 ]]; then
        return 1
    fi

    local open_time=$(printf '%02d%02d' $MARKET_OPEN_H $MARKET_OPEN_M)
    local close_time=$(printf '%02d%02d' $MARKET_CLOSE_H $MARKET_CLOSE_M)

    if [[ "10#$now_et" -ge "10#$open_time" && "10#$now_et" -lt "10#$close_time" ]]; then
        return 0
    else
        return 1
    fi
}

time_until_open() {
    local now_et
    now_et=$(TZ="America/New_York" date '+%H %M %u')
    local hour=$(echo "$now_et" | awk '{print $1}')
    local min=$(echo "$now_et" | awk '{print $2}')
    local dow=$(echo "$now_et" | awk '{print $3}')

    hour=$((10#$hour))
    min=$((10#$min))

    local now_min=$((hour * 60 + min))
    local open_min=$((MARKET_OPEN_H * 60 + MARKET_OPEN_M))

    if [[ $dow -ge 6 ]]; then
        local days_until_mon=$((8 - dow))
        echo $(( (days_until_mon * 24 * 60 - now_min + open_min) * 60 ))
    elif [[ $now_min -lt $open_min ]]; then
        echo $(( (open_min - now_min) * 60 ))
    else
        if [[ $dow -eq 5 ]]; then
            echo $(( (3 * 24 * 60 - now_min + open_min) * 60 ))
        else
            echo $(( (24 * 60 - now_min + open_min) * 60 ))
        fi
    fi
}

# Map strategy name to runner script and data source
get_runner_cmd() {
    local strategy="$1"
    local output="$2"

    case "$strategy" in
        bollinger)
            echo "python strategies/run_bollinger.py --data-path '$DATA_PATH' --watchlist api_data/watchlist.csv --watchlist-mode filter --mode live --skip-live --lookback-days 1 --summary-only $NO_NOTIFY --output-signals $output"
            ;;
        oversold_bounce)
            echo "python strategies/run_oversold_bounce.py --data-path '$DATA_PATH' --watchlist api_data/watchlist.csv --watchlist-mode filter --mode live --skip-live --lookback-days 1 --summary-only $NO_NOTIFY --output-signals $output"
            ;;
        oversold_reversal)
            echo "python strategies/run_oversold_reversal.py --predictions $PREDICTIONS --watchlist api_data/watchlist.csv --watchlist-mode filter --mode live --skip-live --lookback-days 1 --summary-only $NO_NOTIFY --output-signals $output"
            ;;
        low_bounce)
            echo "python strategies/run_low_bounce.py --data-path '$DATA_PATH' --watchlist api_data/watchlist.csv --watchlist-mode filter --mode live --skip-live --lookback-days 1 --summary-only $NO_NOTIFY --output-signals $output"
            ;;
        regression_momentum)
            echo "python strategies/run_regression_momentum.py --predictions $PREDICTIONS --watchlist api_data/watchlist.csv --watchlist-mode filter --mode live --skip-live --lookback-days 1 --summary-only $NO_NOTIFY --output-signals $output"
            ;;
        *)
            log "WARNING: Unknown strategy '$strategy', skipping"
            echo ""
            ;;
    esac
}

merge_signal_files() {
    # Merge per-strategy signal CSVs into a single pending_orders.csv
    # Preserves header from first file, appends data rows from the rest
    local output="$1"
    shift
    local files=("$@")

    local header_written=false
    local total=0

    for f in "${files[@]}"; do
        if [[ ! -f "$f" ]]; then
            continue
        fi

        local lines=$(wc -l < "$f")
        if [[ $lines -le 1 ]]; then
            rm -f "$f"
            continue
        fi

        if [[ "$header_written" == false ]]; then
            head -1 "$f" > "$output"
            header_written=true
        fi

        # Append data rows (skip header)
        tail -n +2 "$f" >> "$output"
        local data_lines=$((lines - 1))
        total=$((total + data_lines))

        rm -f "$f"
    done

    echo "$total"
}

run_scan() {
    local scan_start=$(date +%s)

    log "SCAN START — strategies=[${STRATEGIES}], interval=${INTERVAL_MIN}min"

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN — would execute:"
        log "  python -m api_data.pull_api_data_rt --bulk --predictions $PREDICTIONS --data-dir $DATA_DIR"
        for strategy in $STRATEGIES; do
            local cmd=$(get_runner_cmd "$strategy" "$SIGNAL_DIR/signals_${strategy}.csv")
            if [[ -n "$cmd" ]]; then
                log "  $cmd"
            fi
        done
        return 0
    fi

    # STEP 0: Update data files with bulk RT quotes (1 API call per 100 symbols)
    # This persists fresh prices to all_data_*.csv AND merged_predictions.csv
    # so all strategies see the latest adjusted_close, high, low, volume.
    log "  Updating data files with bulk RT quotes..."
    if python -m api_data.pull_api_data_rt \
        --bulk \
        --predictions "$PREDICTIONS" \
        --data-dir "$DATA_DIR" \
        >> "$LOG_FILE" 2>&1; then
        log "  Bulk RT update complete"
    else
        log "  WARNING: Bulk RT update failed (strategies will use stale data)"
    fi

    # Run each strategy, writing signals to per-strategy files
    local signal_files=()
    local failed=0
    local succeeded=0

    for strategy in $STRATEGIES; do
        local per_file="$SIGNAL_DIR/signals_${strategy}.csv"
        local cmd=$(get_runner_cmd "$strategy" "$per_file")

        if [[ -z "$cmd" ]]; then
            continue
        fi

        log "  Running $strategy..."
        if eval "$cmd" >> "$LOG_FILE" 2>&1; then
            succeeded=$((succeeded + 1))
            signal_files+=("$per_file")
        else
            failed=$((failed + 1))
            log "  WARNING: $strategy scan failed"
        fi
    done

    # Merge per-strategy signal files into one
    local total_signals=0
    if [[ ${#signal_files[@]} -gt 0 ]]; then
        total_signals=$(merge_signal_files "$SIGNAL_FILE" "${signal_files[@]}")
    fi

    local scan_end=$(date +%s)
    local elapsed=$(( scan_end - scan_start ))

    if [[ $total_signals -gt 0 ]]; then
        log "SCAN DONE — ${total_signals} signal(s) from ${succeeded} strategies (${elapsed}s, ${failed} failed)"

        # Execute immediately if --execute flag is set
        if [[ "$EXECUTE" == true ]]; then
            log "EXECUTING — sending ${total_signals} signal(s) to IBKR..."
            local ts=$(date '+%Y%m%d_%H%M%S')

            # Archive a copy before execution
            cp "$SIGNAL_FILE" "$SIGNAL_DIR/executed/scan_${ts}.csv" 2>/dev/null || true

            python -m ibkr.execute_signals \
                --signals "$SIGNAL_FILE" \
                $GATEWAY_FLAG \
                --client-id "$CLIENT_ID" \
                $LIVE_FLAG \
                $DEFAULT_SHARES \
                $MARKET_ORDERS \
                $BUY_ONLY \
                >> "$LOG_FILE" 2>&1

            local exec_code=$?
            if [[ $exec_code -eq 0 ]]; then
                log "EXECUTION COMPLETE"
            else
                log "EXECUTION FAILED — exit code $exec_code"
            fi

            # Clean up signal file (already executed)
            rm -f "$SIGNAL_FILE"
        else
            log "Signals written to $SIGNAL_FILE (use --execute to send to IBKR)"
        fi
    else
        log "SCAN DONE — no signals from ${succeeded} strategies (${elapsed}s, ${failed} failed)"
        # Clean up empty signal file
        rm -f "$SIGNAL_FILE"
    fi

    return 0
}

# ── Banner ──────────────────────────────────────────────────────────
log "============================================"
log "RT SCAN LOOP (multi-strategy)"
log "  Interval:    ${INTERVAL_MIN} min"
log "  Strategies:  ${STRATEGIES}"
log "  Predictions: ${PREDICTIONS}"
log "  Data dir:    ${DATA_DIR}"
log "  Output:      ${SIGNAL_FILE}"
log "  Log:         ${LOG_FILE}"
log "  Dry run:     ${DRY_RUN}"
log "  Buy only:    ${BUY_ONLY:-off}"
log "  One-shot:    ${ONCE}"
log "============================================"

# ── Single run mode ─────────────────────────────────────────────────
if [[ "$ONCE" == true ]]; then
    run_scan
    exit $?
fi

# ── Main loop ───────────────────────────────────────────────────────
log "Entering scan loop (Ctrl+C to stop)"

while true; do
    if is_market_hours; then
        run_scan || true  # don't exit on scan failure

        log "Sleeping ${INTERVAL_MIN} min until next scan..."
        sleep "$INTERVAL_SEC"
    else
        log "Market closed. Shutting down scan loop."
        exit 0
    fi
done
