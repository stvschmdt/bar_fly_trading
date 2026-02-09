#!/bin/bash
# Real-time trade executor loop
#
# Polls for a signal file (pending_orders.csv) and executes trades when found.
# Designed to run alongside rt_scan_loop.sh — the scanner writes signals,
# this script picks them up and sends them to IBKR.
#
# Flow:
#   1. Poll every POLL_SEC seconds for SIGNAL_FILE
#   2. When found, execute via ibkr.execute_signals
#   3. Move executed file to signals/executed/ with timestamp
#   4. Resume polling
#
# Usage:
#   ./scripts/rt_execute_loop.sh                       # default: poll every 2s, paper
#   ./scripts/rt_execute_loop.sh --poll 1              # poll every 1 second
#   ./scripts/rt_execute_loop.sh --dry-run             # print what would execute
#   ./scripts/rt_execute_loop.sh --live                # LIVE trading (requires safety gate)
#   ./scripts/rt_execute_loop.sh --once                # execute once and exit (no loop)
#
# Output:
#   signals/executed/orders_YYYYMMDD_HHMMSS.csv  — archive of executed signals
#   logs/rt_execute.log                          — append-only execution log
#
# Requires:
#   - IBKR Gateway running (paper: port 4002, live: port 4001)
#   - SSH tunnel if Gateway is remote

set -e

cd ~/proj/bar_fly_trading

# ── Config ──────────────────────────────────────────────────────────
POLL_SEC=2                                           # seconds between file checks
SIGNAL_DIR="signals"
SIGNAL_FILE="$SIGNAL_DIR/pending_orders.csv"
EXECUTED_DIR="$SIGNAL_DIR/executed"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/rt_execute.log"
DRY_RUN=false
ONCE=false
LIVE_FLAG=""
CLIENT_ID=10
GATEWAY_FLAG="--gateway"
DEFAULT_SHARES=""      # --default-shares N: fixed share count for testing
MARKET_ORDERS=""       # --market-orders: use market orders instead of limit
BUY_ONLY=""            # --buy-only: skip SELL signals for symbols we don't hold

# Market hours in ET (24h format) — only execute during market hours
MARKET_OPEN_H=9
MARKET_OPEN_M=30
MARKET_CLOSE_H=16
MARKET_CLOSE_M=0

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll)        POLL_SEC="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --once)        ONCE=true; shift ;;
        --live)        LIVE_FLAG="--live"; GATEWAY_FLAG=""; shift ;;
        --default-shares) DEFAULT_SHARES="--default-shares $2"; shift 2 ;;
        --market-orders) MARKET_ORDERS="--market-orders"; shift ;;
        --buy-only)    BUY_ONLY="--buy-only"; shift ;;
        --client-id)   CLIENT_ID="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Setup ───────────────────────────────────────────────────────────
mkdir -p "$SIGNAL_DIR" "$EXECUTED_DIR" "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

is_market_hours() {
    local now_et
    now_et=$(TZ="America/New_York" date '+%H%M')
    local day_of_week
    day_of_week=$(TZ="America/New_York" date '+%u')

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

has_signals() {
    # Check if signal file exists and has data rows (not just header)
    if [[ ! -f "$SIGNAL_FILE" ]]; then
        return 1
    fi

    local lines
    lines=$(wc -l < "$SIGNAL_FILE")
    if [[ $lines -le 1 ]]; then
        # Empty or header-only
        return 1
    fi

    return 0
}

execute_signals() {
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local signal_count=$(( $(wc -l < "$SIGNAL_FILE") - 1 ))

    log "SIGNALS FOUND — ${signal_count} order(s) in $SIGNAL_FILE"

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN — would execute:"
        log "  python -m ibkr.execute_signals \\"
        log "    --signals $SIGNAL_FILE \\"
        log "    $GATEWAY_FLAG --client-id $CLIENT_ID $LIVE_FLAG"
        log ""
        log "Signal contents:"
        cat "$SIGNAL_FILE" >> "$LOG_FILE"
        cat "$SIGNAL_FILE"

        # Archive the file even in dry-run so we don't re-trigger
        mv "$SIGNAL_FILE" "$EXECUTED_DIR/orders_${timestamp}_dryrun.csv"
        log "Archived to $EXECUTED_DIR/orders_${timestamp}_dryrun.csv"
        return 0
    fi

    # Copy signal file before execution (in case executor modifies it)
    cp "$SIGNAL_FILE" "$EXECUTED_DIR/orders_${timestamp}.csv"
    log "Archived copy to $EXECUTED_DIR/orders_${timestamp}.csv"

    # Execute trades
    log "EXECUTING..."
    python -m ibkr.execute_signals \
        --signals "$SIGNAL_FILE" \
        $GATEWAY_FLAG \
        --client-id "$CLIENT_ID" \
        $LIVE_FLAG \
        $DEFAULT_SHARES \
        $MARKET_ORDERS \
        $BUY_ONLY \
        >> "$LOG_FILE" 2>&1

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log "EXECUTION SUCCESS — ${signal_count} order(s) sent"
    else
        log "EXECUTION FAILED — exit code $exit_code"
    fi

    # Remove the signal file so we don't re-execute
    rm -f "$SIGNAL_FILE"
    log "Removed $SIGNAL_FILE (executed)"

    return $exit_code
}

# ── Banner ──────────────────────────────────────────────────────────
log "============================================"
log "RT EXECUTE LOOP"
log "  Poll interval: ${POLL_SEC}s"
log "  Signal file:   ${SIGNAL_FILE}"
log "  Archive dir:   ${EXECUTED_DIR}"
log "  Log:           ${LOG_FILE}"
log "  Mode:          ${LIVE_FLAG:-paper}"
log "  Gateway:       ${GATEWAY_FLAG:-direct}"
log "  Client ID:     ${CLIENT_ID}"
log "  Dry run:       ${DRY_RUN}"
log "  One-shot:      ${ONCE}"
log "============================================"

# ── Single run mode ─────────────────────────────────────────────────
if [[ "$ONCE" == true ]]; then
    if has_signals; then
        execute_signals
        exit $?
    else
        log "No signals found in $SIGNAL_FILE"
        exit 0
    fi
fi

# ── Main loop ───────────────────────────────────────────────────────
log "Polling for signals (Ctrl+C to stop)"

poll_count=0
last_status_time=$(date +%s)
STATUS_INTERVAL=300  # print "still polling" every 5 min

while true; do
    if ! is_market_hours; then
        log "Market closed. Shutting down executor loop."
        exit 0
    fi

    if has_signals; then
        execute_signals || true  # don't exit on execution failure
    fi

    poll_count=$((poll_count + 1))

    # Periodic heartbeat so you know it's alive
    now=$(date +%s)
    if [[ $((now - last_status_time)) -ge $STATUS_INTERVAL ]]; then
        log "Still polling... (${poll_count} checks, no signals)"
        poll_count=0
        last_status_time=$now
    fi

    sleep "$POLL_SEC"
done
