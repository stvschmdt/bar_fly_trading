#!/bin/bash
# Real-time Bollinger scan loop
#
# Runs the bollinger shadow strategy in realtime mode on a repeating interval
# during market hours (9:30am - 4:00pm ET, Mon-Fri).
# Each scan fetches live Alpha Vantage data and writes signals to a CSV file
# that the executor loop (rt_execute_loop.sh) picks up.
#
# Usage:
#   ./scripts/rt_scan_loop.sh                          # default: 15 min interval
#   ./scripts/rt_scan_loop.sh --interval 5             # every 5 minutes
#   ./scripts/rt_scan_loop.sh --interval 1             # every 1 minute
#   ./scripts/rt_scan_loop.sh --dry-run                # print what would run, don't execute
#   ./scripts/rt_scan_loop.sh --once                   # run once and exit (no loop)
#
# Output:
#   signals/pending_orders.csv  — written when signals are found
#   logs/rt_scan.log            — append-only log of all scan runs
#
# Requires:
#   - Alpha Vantage API key (ALPHAVANTAGE_API_KEY env var)
#   - api_data/watchlist.csv

set -e

cd ~/proj/bar_fly_trading

# ── Config ──────────────────────────────────────────────────────────
INTERVAL_MIN=15                                      # minutes between scans
SIGNAL_DIR="signals"
SIGNAL_FILE="$SIGNAL_DIR/pending_orders.csv"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/rt_scan.log"
WATCHLIST="api_data/watchlist.csv"
DRY_RUN=false
ONCE=false

# Market hours in ET (24h format)
MARKET_OPEN_H=9
MARKET_OPEN_M=30
MARKET_CLOSE_H=16
MARKET_CLOSE_M=0

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)    INTERVAL_MIN="$2"; shift 2 ;;
        --watchlist)   WATCHLIST="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --once)        ONCE=true; shift ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
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
    # Check if current time is within market hours (ET)
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

    if [[ "$now_et" -ge "$open_time" && "$now_et" -lt "$close_time" ]]; then
        return 0
    else
        return 1
    fi
}

time_until_open() {
    # Returns seconds until next market open, or 0 if market is open
    local now_et
    now_et=$(TZ="America/New_York" date '+%H %M %u')
    local hour=$(echo "$now_et" | awk '{print $1}')
    local min=$(echo "$now_et" | awk '{print $2}')
    local dow=$(echo "$now_et" | awk '{print $3}')

    # Remove leading zeros for arithmetic
    hour=$((10#$hour))
    min=$((10#$min))

    local now_min=$((hour * 60 + min))
    local open_min=$((MARKET_OPEN_H * 60 + MARKET_OPEN_M))

    if [[ $dow -ge 6 ]]; then
        # Weekend: calculate to Monday 9:30
        local days_until_mon=$((8 - dow))  # Sat=6→2, Sun=7→1
        echo $(( (days_until_mon * 24 * 60 - now_min + open_min) * 60 ))
    elif [[ $now_min -lt $open_min ]]; then
        # Before open today
        echo $(( (open_min - now_min) * 60 ))
    else
        # After close, wait until tomorrow (or Monday)
        if [[ $dow -eq 5 ]]; then
            # Friday after close → Monday open
            echo $(( (3 * 24 * 60 - now_min + open_min) * 60 ))
        else
            echo $(( (24 * 60 - now_min + open_min) * 60 ))
        fi
    fi
}

run_scan() {
    local scan_start=$(date +%s)

    log "SCAN START — interval=${INTERVAL_MIN}min"

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN — would execute:"
        log "  python strategies/bollinger_shadow_strategy.py \\"
        log "    --mode realtime \\"
        log "    --watchlist $WATCHLIST --watchlist-mode filter \\"
        log "    --skip-live --summary-only \\"
        log "    --output-signals $SIGNAL_FILE"
        return 0
    fi

    # Run the realtime scan
    python strategies/bollinger_shadow_strategy.py \
        --mode realtime \
        --watchlist "$WATCHLIST" --watchlist-mode filter \
        --skip-live --summary-only \
        --output-signals "$SIGNAL_FILE" \
        >> "$LOG_FILE" 2>&1

    local exit_code=$?
    local scan_end=$(date +%s)
    local elapsed=$(( scan_end - scan_start ))

    if [[ $exit_code -eq 0 ]]; then
        if [[ -f "$SIGNAL_FILE" ]]; then
            local signal_count=$(( $(wc -l < "$SIGNAL_FILE") - 1 ))  # subtract header
            log "SCAN DONE — ${signal_count} signal(s) written to $SIGNAL_FILE (${elapsed}s)"
        else
            log "SCAN DONE — no signals (${elapsed}s)"
        fi
    else
        log "SCAN FAILED — exit code $exit_code (${elapsed}s)"
    fi

    return $exit_code
}

# ── Banner ──────────────────────────────────────────────────────────
log "============================================"
log "RT SCAN LOOP"
log "  Interval:  ${INTERVAL_MIN} min"
log "  Watchlist: ${WATCHLIST}"
log "  Output:    ${SIGNAL_FILE}"
log "  Log:       ${LOG_FILE}"
log "  Dry run:   ${DRY_RUN}"
log "  One-shot:  ${ONCE}"
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
        wait_sec=$(time_until_open)
        wait_min=$((wait_sec / 60))
        wait_hr=$((wait_min / 60))
        remaining_min=$((wait_min % 60))

        log "Market closed. Next open in ~${wait_hr}h ${remaining_min}m. Sleeping..."

        # Sleep in chunks so we can be interrupted cleanly
        # Check every 5 min if market has opened (handles DST, holidays loosely)
        local_sleep=$((wait_sec < 300 ? wait_sec : 300))
        sleep "$local_sleep"
    fi
done
