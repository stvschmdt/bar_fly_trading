# IBKR Gateway Test Suite

## Remote Access Setup

The IB Gateway runs on AWS (54.90.246.184). Use SSH tunnel for remote access:

```bash
# Start SSH tunnel (run once, keeps running in background)
ssh -L 4001:127.0.0.1:4001 sschmidt@54.90.246.184 -N &
```

## Quick Start

```bash
cd ~/proj/ibkr
python test_gateway.py --balance    # Verify connection works, shows account value
```

## Commands

| Command | Purpose | What to Look For |
|---------|---------|------------------|
| `python test_gateway.py --balance` | Check all account balances | Net Liquidation > 0, Available Funds > 0 |
| `python test_gateway.py --balance --account U17955245` | Check specific account | Single account balance |
| `python test_gateway.py --portfolio` | List current positions | Table of holdings with shares, cost, P&L |
| `python test_gateway.py --all` | Run balance + portfolio | Both tests pass |
| `python test_gateway.py --buy --symbol NKE` | Buy 1 share of NKE | Order ID assigned, status Filled or Submitted |
| `python test_gateway.py --sell --symbol NKE` | Sell 1 share of NKE | Requires existing position, status Filled |
| `python test_gateway.py --buy --symbol AAPL --shares 2` | Buy 2 shares of AAPL | Custom symbol/quantity |
| `python test_gateway.py --buy --symbol NKE --dry-run` | Simulate buy (no trade) | Shows price, skips order submission |

## Expected Output

**Success:**
```
SUCCESS: Connected to IB Gateway
Net Liquidation Value: $xxx,xxx.xx
SUCCESS: Account balance retrieved
```

**Failure (connection):**
```
ERROR: Failed to connect to Gateway at 127.0.0.1:4001
```
Make sure SSH tunnel is running.

**After Hours (buy/sell):**
```
Order not filled (status: Inactive)
```
This is normal - market orders don't fill outside 9:30 AM - 4:00 PM ET.

## Flags Reference

- `--host` - Gateway IP (default: 127.0.0.1 via SSH tunnel)
- `--port` - Gateway port (default: 4001 = live)
- `--symbol` - Stock ticker (default: NKE)
- `--shares` - Quantity (default: 1)
- `--dry-run` - Don't execute trade, just show what would happen
- `--client-id` - IBKR client ID (default: 1)
- `--account` - Specific account ID (default: show all accounts)

## Multi-Account Support

FA accounts show multiple managed accounts:
- **F* accounts** (e.g., F15468824) - FA Master account
- **U* accounts** (e.g., U17955245) - Individual client accounts

By default, `--balance` shows all accounts. Use `--account U17955245` to filter to one.

## AWS Server Details

- **Public IP:** 54.90.246.184
- **Gateway Port:** 4001 (live trading)
- **Service:** `sudo systemctl status ibgateway`
- **Logs:** `/home/ibkr/ibc/logs/`