#!/bin/bash
# Deploy BFT webapp to EC2 production server
#
# Usage (run on the EC2 server):
#   cd ~/bar_fly_trading
#   bash scripts/deploy_ec2.sh              # full deploy
#   bash scripts/deploy_ec2.sh --frontend   # frontend only
#   bash scripts/deploy_ec2.sh --backend    # backend restart only
#   bash scripts/deploy_ec2.sh --data       # repopulate data only

set -e

REPO_DIR="$HOME/bar_fly_trading"
DATA_DIR="/var/www/bft/data"
FRONTEND_DIR="/var/www/bft/frontend"
LOG_DIR="/var/log/bft"
CSV_PATTERN="$REPO_DIR/all_data_*.csv"
BRANCH="feature/website"

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ── Parse args ────────────────────────────────────────────────────
DO_ALL=true
DO_PULL=false
DO_FRONTEND=false
DO_BACKEND=false
DO_DATA=false

if [[ $# -gt 0 ]]; then
    DO_ALL=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --frontend)  DO_FRONTEND=true; DO_PULL=true; shift ;;
            --backend)   DO_BACKEND=true; DO_PULL=true; shift ;;
            --data)      DO_DATA=true; shift ;;
            --pull)      DO_PULL=true; shift ;;
            *)           echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
fi

# ── Ensure directories exist ─────────────────────────────────────
sudo mkdir -p "$DATA_DIR" "$FRONTEND_DIR" "$LOG_DIR"
sudo chown -R "$USER:$USER" /var/www/bft "$LOG_DIR" 2>/dev/null || true

# ── Pull latest code ─────────────────────────────────────────────
if $DO_ALL || $DO_PULL; then
    log "Pulling latest from $BRANCH..."
    git pull origin "$BRANCH"
fi

# ── Frontend ──────────────────────────────────────────────────────
if $DO_ALL || $DO_FRONTEND; then
    log "Building frontend..."
    cd "$REPO_DIR/webapp/frontend"
    npm install --silent 2>/dev/null
    npm run build
    cp -r dist/* "$FRONTEND_DIR/"
    log "Frontend deployed to $FRONTEND_DIR"
    cd "$REPO_DIR"
fi

# ── Python deps ───────────────────────────────────────────────────
if $DO_ALL; then
    log "Installing Python dependencies..."
    pip install -q fastapi uvicorn aiofiles bcrypt pyjwt python-multipart 2>/dev/null || true
fi

# ── Data population ───────────────────────────────────────────────
if $DO_ALL || $DO_DATA; then
    log "Building sector map..."
    BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.build_sector_map --csv-pattern "$CSV_PATTERN"

    log "Populating symbol data from CSVs..."
    BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.populate_all --csv-pattern "$CSV_PATTERN"

    # Create default invite code if DB doesn't exist yet
    if [ ! -f "$DATA_DIR/bft_auth.db" ]; then
        log "Creating invite code BETA2026..."
        BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.database BETA2026 20
    fi
fi

# ── Backend ───────────────────────────────────────────────────────
if $DO_ALL || $DO_BACKEND; then
    log "Restarting backend..."

    # Kill existing uvicorn if running
    pkill -f "uvicorn webapp.backend.api:app" 2>/dev/null || true
    sleep 1

    BFT_DATA_DIR="$DATA_DIR" nohup python -m uvicorn webapp.backend.api:app \
        --host 127.0.0.1 --port 8000 >> "$LOG_DIR/api.log" 2>&1 &

    sleep 2

    # Verify
    if curl -sf http://localhost:8000/api/health > /dev/null; then
        log "Backend running (PID $!)"
    else
        log "ERROR: Backend failed to start. Check $LOG_DIR/api.log"
        tail -10 "$LOG_DIR/api.log"
        exit 1
    fi
fi

# ── Nginx reload ──────────────────────────────────────────────────
if $DO_ALL || $DO_FRONTEND; then
    sudo nginx -t && sudo systemctl reload nginx
    log "Nginx reloaded"
fi

log "Deploy complete! Visit https://www.barflytrading.com"
