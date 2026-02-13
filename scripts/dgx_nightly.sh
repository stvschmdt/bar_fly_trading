#!/bin/bash
# DGX Nightly — AI summaries via ollama, sync to EC2
#
# Runs after ec2_nightly.sh finishes. Pulls symbol JSONs from EC2,
# runs generate_reports (ollama AI summaries), pushes updated JSONs back.
#
# Usage:
#   bash scripts/dgx_nightly.sh                         # full: pull + reports + push
#   bash scripts/dgx_nightly.sh --step pull              # just pull JSONs from EC2
#   bash scripts/dgx_nightly.sh --step reports           # just run generate_reports
#   bash scripts/dgx_nightly.sh --step push              # just push JSONs to EC2
#
# Cron (7pm ET weekdays — 1 hour after EC2 nightly):
#   0 19 * * 1-5 cd ~/proj/bar_fly_trading && bash scripts/dgx_nightly.sh >> logs/dgx_nightly.log 2>&1

set -o pipefail

REPO_DIR="$HOME/proj/bar_fly_trading"
LOCAL_DATA_DIR="$REPO_DIR/webapp_data"
EC2_HOST="${EC2_HOST:?Set EC2_HOST env var (e.g. user@ip)}"
EC2_DATA_DIR="/var/www/bft/data"
CSV_PATTERN="${REPO_DIR}/all_data_*.csv"

cd "$REPO_DIR" || exit 1
mkdir -p "$LOCAL_DATA_DIR" logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── GPU conflict detection ───────────────────────────────────────────
# If model training is using the GPU, ollama will OOM.
# Detect this and set BFT_OLLAMA_MODEL=None to skip LLM summaries,
# or fall back to a different provider via env var.
check_gpu_available() {
    # Check if any python training processes are using the GPU
    if nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null | grep -qi python; then
        log "  GPU in use by training — falling back to CPU model"
        # Use smallest model that can run on CPU alongside training
        export BFT_OLLAMA_MODEL="llama3.2:1b"
        return 1
    fi
    return 0
}

# ── Parse args ───────────────────────────────────────────────────────
STEP="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) STEP="$2"; shift 2 ;;
        *)      log "Unknown arg: $1"; exit 1 ;;
    esac
done

FAILED=0

# ── STEP 1: Pull symbol JSONs from EC2 ──────────────────────────────
run_pull() {
    log "=== STEP 1: Pull symbol JSONs from EC2 ==="
    if scp "$EC2_HOST:$EC2_DATA_DIR/*.json" "$LOCAL_DATA_DIR/" 2>&1; then
        local count=$(ls "$LOCAL_DATA_DIR"/*.json 2>/dev/null | wc -l)
        log "  Pulled $count JSON files"
    else
        log "  ERROR: scp pull failed"
        FAILED=$((FAILED + 1))
    fi
}

# ── STEP 2: Run generate_reports (ollama AI summaries) ───────────────
run_reports() {
    log "=== STEP 2: Generate AI reports ==="

    check_gpu_available
    # BFT_OLLAMA_MODEL is now set: default 3b if GPU free, 1b if busy

    # Ensure ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "  Starting ollama..."
        ollama serve &
        sleep 3
    fi

    export BFT_OLLAMA_MODEL="${BFT_OLLAMA_MODEL:-llama3.1:8b}"
    log "  Running with ollama ($BFT_OLLAMA_MODEL)"
    if BFT_DATA_DIR="$LOCAL_DATA_DIR" BFT_OLLAMA_MODEL="$BFT_OLLAMA_MODEL" python -m webapp.backend.generate_reports \
        --csv-pattern "$CSV_PATTERN" 2>&1; then
        log "  Reports generated with AI summaries"
    else
        log "  WARNING: generate_reports had errors (some summaries may be missing)"
        FAILED=$((FAILED + 1))
    fi
}

# ── STEP 3: Push updated JSONs back to EC2 ──────────────────────────
run_push() {
    log "=== STEP 3: Push updated JSONs to EC2 ==="
    if scp "$LOCAL_DATA_DIR"/*.json "$EC2_HOST:$EC2_DATA_DIR/" 2>&1; then
        local count=$(ls "$LOCAL_DATA_DIR"/*.json 2>/dev/null | wc -l)
        log "  Pushed $count JSON files to EC2"
    else
        log "  ERROR: scp push failed"
        FAILED=$((FAILED + 1))
    fi
}

# ── Dispatch ─────────────────────────────────────────────────────────
log "============================================"
log "DGX NIGHTLY (step=$STEP)"
log "  EC2: $EC2_HOST:$EC2_DATA_DIR"
log "  Local: $LOCAL_DATA_DIR"
log "============================================"

case "$STEP" in
    all)     run_pull; run_reports; run_push ;;
    pull)    run_pull ;;
    reports) run_reports ;;
    push)    run_push ;;
    *)       log "Unknown step: $STEP (use: pull, reports, push, or all)"; exit 1 ;;
esac

if [ $FAILED -gt 0 ]; then
    log "DONE with $FAILED warning(s)/error(s)"
else
    log "DONE — all steps succeeded"
fi
