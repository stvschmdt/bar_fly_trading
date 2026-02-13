#!/bin/bash
# EC2 Market Hours — RT scan + execute loop
#
# Wraps rt_scan_loop.sh with EC2-specific paths and options.
# Runs from 9:30am-4pm ET, scanning every 15 min, executing trades.
#
# Usage:
#   bash scripts/ec2_market_hours.sh                    # default: paper, 15 min
#   bash scripts/ec2_market_hours.sh --live              # real money
#   bash scripts/ec2_market_hours.sh --interval 5        # every 5 min
#   bash scripts/ec2_market_hours.sh --dry-run           # print what would run
#   bash scripts/ec2_market_hours.sh --once              # single scan, then exit
#
# Cron (start at market open, loop handles close):
#   30 9 * * 1-5 cd /home/sschmidt/bar_fly_trading && bash scripts/ec2_market_hours.sh >> /var/log/bft/market_hours.log 2>&1
#
# Monitor:
#   tail -f /var/log/bft/market_hours.log
#   tail -f logs/rt_scan.log

REPO_DIR="${HOME}/bar_fly_trading"
DATA_DIR="${REPO_DIR}"
LOG_DIR="/var/log/bft"

cd "$REPO_DIR" || exit 1
mkdir -p "$LOG_DIR" logs signals/executed

# Forward all args to rt_scan_loop.sh
# EC2 defaults: execute trades, buy-only, paper mode, gateway
exec bash scripts/rt_scan_loop.sh \
    --data-dir "$DATA_DIR" \
    --execute \
    --buy-only \
    # --no-notify \
    "$@"
