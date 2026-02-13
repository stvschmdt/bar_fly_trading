#!/bin/bash
# Lightweight intraday refresh: re-reads all_data CSVs and updates website JSON.
# Intended to run every 15 min during market hours via cron.
# Does NOT call any external APIs or run LLM summaries.

REPO_DIR="/home/sschmidt/bar_fly_trading"
DATA_DIR="/var/www/bft/data"
CSV_PATTERN="$REPO_DIR/all_data_*.csv"

cd "$REPO_DIR" || exit 1

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "Refreshing website data from CSVs..."

BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.build_sector_map --csv-pattern "$CSV_PATTERN" --output-dir "$DATA_DIR" \
    && log "Sector map updated" || log "WARNING: build_sector_map failed"

BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.populate_all --csv-pattern "$CSV_PATTERN" \
    && log "Symbol data updated" || log "WARNING: populate_all failed"

BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.generate_history --csv-pattern "$CSV_PATTERN" \
    && log "Chart history updated" || log "WARNING: generate_history failed"

log "Done"
