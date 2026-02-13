#!/bin/bash
# EC2 Nightly Pipeline
#
# Runs after market close. Pulls fresh data, builds gold tables,
# generates screener PDF, uploads to Drive, refreshes website data.
#
# Usage:
#   bash scripts/ec2_nightly.sh              # full pipeline
#   bash scripts/ec2_nightly.sh --step data  # just pull_api_data + gold tables
#   bash scripts/ec2_nightly.sh --step pdf   # just screener PDF + upload
#   bash scripts/ec2_nightly.sh --step web   # just website data refresh
#
# Cron (6pm ET weekdays):
#   0 18 * * 1-5 cd /home/sschmidt/bar_fly_trading && bash scripts/ec2_nightly.sh >> /var/log/bft/nightly.log 2>&1

set -o pipefail

REPO_DIR="${HOME}/bar_fly_trading"
DATA_DIR="/var/www/bft/data"
CSV_PATTERN="${REPO_DIR}/all_data_*.csv"
LOG_DIR="/var/log/bft"
LOCK_FILE="/tmp/ec2_nightly.lock"

cd "$REPO_DIR" || exit 1
mkdir -p "$LOG_DIR" logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── Lock ─────────────────────────────────────────────────────────────
if [ -f "$LOCK_FILE" ]; then
    log "ERROR: Lock file exists ($LOCK_FILE). Already running or stale lock."
    log "  To force: rm $LOCK_FILE"
    exit 1
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

# ── Parse args ───────────────────────────────────────────────────────
STEP="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) STEP="$2"; shift 2 ;;
        *)      log "Unknown arg: $1"; exit 1 ;;
    esac
done

FAILED=0

# ── STEP 1: Pull API data + gold tables ─────────────────────────────
run_data() {
    log "=== STEP 1: Pull API data + gold tables ==="
    if python -m api_data.pull_api_data -w all 2>&1; then
        log "  Data pull complete"
    else
        log "  ERROR: Data pull failed"
        FAILED=$((FAILED + 1))
    fi
}

# ── STEP 2: Screener PDF + upload to Drive ──────────────────────────
run_pdf() {
    log "=== STEP 2: Screener PDF + Drive upload ==="
    if python -m visualizations.screener_v2 --n_days 60 --data "$REPO_DIR" 2>&1; then
        log "  Screener PDF generated"
    else
        log "  ERROR: Screener PDF failed"
        FAILED=$((FAILED + 1))
        return
    fi

    # Upload to Google Drive
    local pdf=$(ls -t overnight_*.pdf 2>/dev/null | head -1)
    if [ -n "$pdf" ]; then
        if python -c "
from cron import upload_to_drive
upload_to_drive('$pdf', '$pdf')
" 2>&1; then
            log "  Uploaded $pdf to Drive"
        else
            log "  WARNING: Drive upload failed"
            FAILED=$((FAILED + 1))
        fi
        # Cleanup (keep overnight_v2_* dirs — webapp serves sector analysis images from them)
        rm -f overnight_*.pdf screener_results_*.csv table_image.jpg 2>/dev/null
    else
        log "  WARNING: No PDF generated"
        FAILED=$((FAILED + 1))
    fi
}

# ── STEP 3: Refresh website data (sector map, symbols, history) ─────
run_web() {
    log "=== STEP 3: Website data refresh ==="

    log "  Building sector map..."
    if BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.build_sector_map \
        --csv-pattern "$CSV_PATTERN" --output-dir "$DATA_DIR" 2>&1; then
        log "  Sector map updated"
    else
        log "  WARNING: build_sector_map failed"
        FAILED=$((FAILED + 1))
    fi

    log "  Populating symbol data..."
    if BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.populate_all \
        --csv-pattern "$CSV_PATTERN" 2>&1; then
        log "  Symbol data updated"
    else
        log "  WARNING: populate_all failed"
        FAILED=$((FAILED + 1))
    fi

    log "  Generating chart history..."
    if BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.generate_history \
        --csv-pattern "$CSV_PATTERN" 2>&1; then
        log "  Chart history updated"
    else
        log "  WARNING: generate_history failed"
        FAILED=$((FAILED + 1))
    fi
}

# ── Dispatch ─────────────────────────────────────────────────────────
log "============================================"
log "EC2 NIGHTLY PIPELINE (step=$STEP)"
log "============================================"

case "$STEP" in
    all)  run_data; run_pdf; run_web ;;
    data) run_data ;;
    pdf)  run_pdf ;;
    web)  run_web ;;
    *)    log "Unknown step: $STEP (use: data, pdf, web, or all)"; exit 1 ;;
esac

if [ $FAILED -gt 0 ]; then
    log "DONE with $FAILED warning(s)/error(s)"
else
    log "DONE — all steps succeeded"
fi
