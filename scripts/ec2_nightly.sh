#!/bin/bash
# EC2 Nightly Pipeline
#
# Full chain (cron default):
#   1. Data pull  — Alpha Vantage → MySQL → all_data_*.csv  (~4 hrs)
#   2. Screener   — overnight charts + PDF → Drive upload    (~15 min)
#   3. SCP → DGX  — send all_data_*.csv to DGX              (~2 min)
#   4. Inference  — SSH DGX: 9 models → merged_predictions   (~2 min)
#   5. SCP ← DGX  — pull merged_predictions.csv back        (~1 min)
#   6. Web refresh — sector map, symbols, history JSONs      (~10 min)
#
# Groups (run independently if chain breaks):
#   bash scripts/ec2_nightly.sh --step ec2      # steps 1,2,3  (data + screener + SCP to DGX)
#   bash scripts/ec2_nightly.sh --step dgx      # steps 4,5    (inference + SCP back)
#   bash scripts/ec2_nightly.sh --step web      # step  6      (website data refresh)
#
# Individual steps:
#   bash scripts/ec2_nightly.sh --step data     # just pull_api_data + gold tables
#   bash scripts/ec2_nightly.sh --step pdf      # just screener PDF + upload
#   bash scripts/ec2_nightly.sh --step scp-to   # just SCP CSVs to DGX
#   bash scripts/ec2_nightly.sh --step infer    # just SSH trigger inference on DGX
#   bash scripts/ec2_nightly.sh --step scp-back # just SCP predictions from DGX
#
# Cron (6pm ET weekdays):
#   0 18 * * 1-5 cd /home/sschmidt/bar_fly_trading && bash scripts/ec2_nightly.sh >> /var/log/bft/nightly.log 2>&1

set -o pipefail

REPO_DIR="${HOME}/bar_fly_trading"
DATA_DIR="/var/www/bft/data"
CSV_PATTERN="${REPO_DIR}/all_data_*.csv"
LOG_DIR="/var/log/bft"
LOCK_FILE="/tmp/ec2_nightly.lock"

# DGX connection (Tailscale)
DGX_HOST="stvschmdt@100.115.147.21"
DGX_REPO="~/proj/bar_fly_trading"
DGX_SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"

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

# ══════════════════════════════════════════════════════════════════════
# GROUP 1: EC2 local (data + screener + SCP to DGX)
# ══════════════════════════════════════════════════════════════════════

# ── Step 1: Pull API data + gold tables ──────────────────────────────
run_data() {
    log "=== STEP 1: Pull API data + gold tables ==="
    if python -m api_data.pull_api_data -w all 2>&1; then
        log "  Data pull complete"
    else
        log "  ERROR: Data pull failed"
        FAILED=$((FAILED + 1))
    fi
}

# ── Step 2: Screener PDF + upload to Drive ───────────────────────────
run_pdf() {
    log "=== STEP 2: Screener PDF + Drive upload ==="
    if python -m visualizations.screener_v2 --n_days 60 --data "$REPO_DIR" 2>&1; then
        log "  Screener charts generated"
    else
        log "  ERROR: Screener failed"
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
        # Cleanup PDFs only — keep overnight_* dirs for webapp sector images
        rm -f overnight_*.pdf screener_results_*.csv table_image.jpg 2>/dev/null
    else
        log "  WARNING: No PDF generated"
        FAILED=$((FAILED + 1))
    fi
}

# ── Step 3: SCP CSVs to DGX ─────────────────────────────────────────
run_scp_to_dgx() {
    log "=== STEP 3: SCP all_data_*.csv → DGX ==="
    if scp $DGX_SSH_OPTS "$REPO_DIR"/all_data_*.csv "${DGX_HOST}:${DGX_REPO}/" 2>&1; then
        log "  Sent $(ls "$REPO_DIR"/all_data_*.csv | wc -l) CSV files to DGX"
    else
        log "  ERROR: SCP to DGX failed"
        FAILED=$((FAILED + 1))
    fi
}

# ══════════════════════════════════════════════════════════════════════
# GROUP 2: DGX inference (trigger remotely via SSH)
# ══════════════════════════════════════════════════════════════════════

# ── Step 4: Run inference on DGX ─────────────────────────────────────
run_inference() {
    log "=== STEP 4: Inference on DGX (9 models) ==="
    if ssh $DGX_SSH_OPTS "$DGX_HOST" "cd ${DGX_REPO} && bash scripts/infer_merge.sh" 2>&1; then
        log "  Inference complete"
    else
        log "  ERROR: DGX inference failed"
        FAILED=$((FAILED + 1))
    fi
}

# ── Step 5: SCP predictions back from DGX ────────────────────────────
run_scp_from_dgx() {
    log "=== STEP 5: SCP merged_predictions.csv ← DGX ==="
    if scp $DGX_SSH_OPTS "${DGX_HOST}:${DGX_REPO}/merged_predictions.csv" "$REPO_DIR/" 2>&1; then
        local size=$(ls -lh "$REPO_DIR/merged_predictions.csv" | awk '{print $5}')
        log "  Received merged_predictions.csv ($size)"
    else
        log "  ERROR: SCP from DGX failed"
        FAILED=$((FAILED + 1))
    fi
}

# ══════════════════════════════════════════════════════════════════════
# GROUP 3: Web refresh (standalone)
# ══════════════════════════════════════════════════════════════════════

# ── Step 6: Refresh website data ─────────────────────────────────────
run_web() {
    log "=== STEP 6: Website data refresh ==="

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
    all)      run_data; run_pdf; run_scp_to_dgx; run_inference; run_scp_from_dgx; run_web ;;
    ec2)      run_data; run_pdf; run_scp_to_dgx ;;
    dgx)      run_inference; run_scp_from_dgx ;;
    data)     run_data ;;
    pdf)      run_pdf ;;
    scp-to)   run_scp_to_dgx ;;
    infer)    run_inference ;;
    scp-back) run_scp_from_dgx ;;
    web)      run_web ;;
    *)        log "Unknown step: $STEP"; log "  Groups: all, ec2, dgx, web"; log "  Steps:  data, pdf, scp-to, infer, scp-back"; exit 1 ;;
esac

if [ $FAILED -gt 0 ]; then
    log "DONE with $FAILED warning(s)/error(s)"
else
    log "DONE — all steps succeeded"
fi
