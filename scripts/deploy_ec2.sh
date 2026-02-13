#!/bin/bash
# Deploy BFT webapp to EC2 production server
#
# Usage (run on the EC2 server):
#   cd ~/bar_fly_trading
#   bash scripts/deploy_ec2.sh              # full deploy
#   bash scripts/deploy_ec2.sh --frontend   # frontend only
#   bash scripts/deploy_ec2.sh --backend    # backend restart only
#   bash scripts/deploy_ec2.sh --data       # repopulate data only
#   bash scripts/deploy_ec2.sh --data --csv-pattern "/other/path/all_data_*.csv"
#   bash scripts/deploy_ec2.sh --setup      # first-time systemd + nginx setup

set -e

REPO_DIR="$HOME/bar_fly_trading"
DATA_DIR="/var/www/bft/data"
FRONTEND_DIR="/var/www/bft/frontend"
LOG_DIR="/var/log/bft"
CSV_PATTERN="${BFT_CSV_PATTERN:-$REPO_DIR/all_data_*.csv}"
BRANCH="feature/ec2-scripts"
SERVICE_NAME="bft-api"
PYTHON_BIN="${CONDA_PREFIX:-$(dirname $(which python))/..}/bin/python"

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ── Parse args ────────────────────────────────────────────────────
DO_ALL=true
DO_PULL=false
DO_FRONTEND=false
DO_BACKEND=false
DO_DATA=false
DO_SETUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --frontend)  DO_ALL=false; DO_FRONTEND=true; DO_PULL=true; shift ;;
        --backend)   DO_ALL=false; DO_BACKEND=true; DO_PULL=true; shift ;;
        --data)      DO_ALL=false; DO_DATA=true; shift ;;
        --pull)      DO_ALL=false; DO_PULL=true; shift ;;
        --setup)     DO_ALL=false; DO_SETUP=true; shift ;;
        --csv-pattern) CSV_PATTERN="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Ensure directories exist ─────────────────────────────────────
sudo mkdir -p "$DATA_DIR" "$FRONTEND_DIR" "$LOG_DIR"
sudo chown -R "$USER:$USER" /var/www/bft "$LOG_DIR" 2>/dev/null || true

# ── First-time setup: systemd service + nginx ─────────────────────
if $DO_SETUP; then
    log "Setting up systemd service..."

    # Generate JWT secret if not already set
    if [ ! -f "$DATA_DIR/.jwt_secret" ]; then
        python3 -c "import secrets; print(secrets.token_urlsafe(32))" > "$DATA_DIR/.jwt_secret"
        chmod 600 "$DATA_DIR/.jwt_secret"
        log "Generated JWT secret at $DATA_DIR/.jwt_secret"
    fi
    JWT_SECRET=$(cat "$DATA_DIR/.jwt_secret")

    # Create env file for secrets (only if it doesn't exist — preserves SMTP vars)
    ENV_FILE="$DATA_DIR/.bft-api.env"
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << ENVEOF
BFT_DATA_DIR=$DATA_DIR
BFT_JWT_SECRET=$JWT_SECRET
# Uncomment and set these for welcome emails:
#IBKR_SMTP_SERVER=smtp.gmail.com
#IBKR_SMTP_PORT=587
#IBKR_SMTP_USER=your@gmail.com
#IBKR_SMTP_PASSWORD=your-app-password
ENVEOF
        chmod 600 "$ENV_FILE"
        log "Created env file at $ENV_FILE — edit to add SMTP credentials"
    else
        # Always update non-secret vars
        sed -i "s|^BFT_DATA_DIR=.*|BFT_DATA_DIR=$DATA_DIR|" "$ENV_FILE"
        sed -i "s|^BFT_JWT_SECRET=.*|BFT_JWT_SECRET=$JWT_SECRET|" "$ENV_FILE"
        log "Env file already exists at $ENV_FILE — SMTP vars preserved"
    fi

    sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=BFT FastAPI Backend
After=network.target

[Service]
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=$PYTHON_BIN -m uvicorn webapp.backend.api:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
EnvironmentFile=$ENV_FILE

[Install]
WantedBy=multi-user.target
EOF

    # Kill any nohup uvicorn still running
    pkill -f "uvicorn webapp.backend.api:app" 2>/dev/null || true
    sleep 1

    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    sudo systemctl start "$SERVICE_NAME"
    log "Systemd service '$SERVICE_NAME' created and started"

    # Nginx config
    if [ ! -f /etc/nginx/conf.d/barflytrading.com.conf ] || ! grep -q proxy_pass /etc/nginx/conf.d/barflytrading.com.conf; then
        log "Nginx config needs API proxy — check /etc/nginx/conf.d/barflytrading.com.conf"
    fi

    sudo systemctl status "$SERVICE_NAME" --no-pager
    exit 0
fi

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
    BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.build_sector_map --csv-pattern "$CSV_PATTERN" --output-dir "$DATA_DIR"

    log "Populating symbol data from CSVs..."
    BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.populate_all --csv-pattern "$CSV_PATTERN"

    log "Generating chart history data..."
    BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.generate_history --csv-pattern "$CSV_PATTERN"

    # NOTE: generate_reports (LLM summaries) runs on DGX via dgx_nightly.sh, not here

    # Create default invite code if DB doesn't exist yet
    if [ ! -f "$DATA_DIR/bft_auth.db" ]; then
        log "Creating invite code BETA2026..."
        BFT_DATA_DIR="$DATA_DIR" python -m webapp.backend.database BETA2026 20
    fi
fi

# ── Backend (restart via systemd) ─────────────────────────────────
if $DO_ALL || $DO_BACKEND; then
    if systemctl is-active "$SERVICE_NAME" > /dev/null 2>&1; then
        log "Restarting backend via systemd..."
        sudo systemctl restart "$SERVICE_NAME"
    else
        log "Starting backend via systemd..."
        sudo systemctl start "$SERVICE_NAME"
    fi

    sleep 2

    if curl -sf http://localhost:8000/api/health > /dev/null; then
        log "Backend running"
    else
        log "ERROR: Backend failed to start"
        sudo journalctl -u "$SERVICE_NAME" --no-pager -n 15
        exit 1
    fi
fi

# ── Nginx reload ──────────────────────────────────────────────────
if $DO_ALL || $DO_FRONTEND; then
    sudo nginx -t && sudo systemctl reload nginx
    log "Nginx reloaded"
fi

log "Deploy complete! Visit https://www.barflytrading.com"
