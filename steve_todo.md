# Steve's Hosting Prep TODO

Step-by-step guide to get the BFT web app hosted at `www.barflytrading.com`.

---

## 1. Domain DNS Setup

- [ ] Log into your domain registrar (Route 53, Namecheap, etc.)
- [ ] Create an **A record** pointing `barflytrading.com` → your EC2 public IP
- [ ] Create a **CNAME** for `www` → `barflytrading.com`
- [ ] Wait for DNS propagation (usually 5-30 min)

## 2. EC2 Instance Prep

- [ ] Confirm your EC2 instance is running (t3.small or larger)
- [ ] Note the **public IP** and **security group ID**
- [ ] Open these ports in the security group inbound rules:
  - `80` (HTTP) — for Let's Encrypt cert challenge + redirect
  - `443` (HTTPS) — for the app
  - `22` (SSH) — already open if you can SSH in
- [ ] SSH into the instance and confirm you have sudo access

## 3. Install System Dependencies (on EC2)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Python 3.12+ (should already be there)
python3 --version

# Node.js 20+ (for building the React frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Nginx
sudo apt install -y nginx

# Certbot (Let's Encrypt SSL)
sudo apt install -y certbot python3-certbot-nginx

# pip packages for the backend
pip install fastapi 'uvicorn[standard]' aiofiles bcrypt pyjwt python-multipart
```

## 4. Clone the Repo & Checkout Website Branch

```bash
cd ~
git clone git@github.com:stvschmdt/bar_fly_trading.git
cd bar_fly_trading
git checkout feature/website
```

## 5. Deploy Directory Setup

```bash
sudo mkdir -p /var/www/bft/{frontend,data}
sudo chown -R $USER:$USER /var/www/bft
sudo mkdir -p /var/log/bft
sudo chown -R $USER:$USER /var/log/bft
```

## 6. Build & Deploy Frontend

```bash
cd ~/bar_fly_trading/webapp/frontend
npm install && npm run build
cp -r dist/* /var/www/bft/frontend/
```

## 7. Populate Data (one-time, from CSVs — no API calls)

```bash
cd ~/bar_fly_trading

# Populate all ~543 symbol JSONs from all_data CSVs (takes ~2 sec)
BFT_DATA_DIR=/var/www/bft/data python -m webapp.backend.populate_all

# Generate invite codes for beta testers
BFT_DATA_DIR=/var/www/bft/data python -m webapp.backend.database BETA2026 20
```

## 8. SSL Certificate (Let's Encrypt)

```bash
sudo certbot --nginx -d barflytrading.com -d www.barflytrading.com
sudo certbot renew --dry-run
```

## 9. Nginx Configuration

Create `/etc/nginx/sites-available/bft`:

```nginx
server {
    listen 80;
    server_name barflytrading.com www.barflytrading.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name barflytrading.com www.barflytrading.com;

    ssl_certificate /etc/letsencrypt/live/barflytrading.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/barflytrading.com/privkey.pem;

    # React frontend (static build)
    root /var/www/bft/frontend;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # FastAPI backend (API only)
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/bft /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

## 10. Systemd Service for FastAPI

Create `/etc/systemd/system/bft-api.service`:

```ini
[Unit]
Description=BFT FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/bar_fly_trading
ExecStart=/usr/bin/python3 -m uvicorn webapp.backend.api:app --host 127.0.0.1 --port 8000
Restart=always
Environment=BFT_DATA_DIR=/var/www/bft/data
Environment=BFT_JWT_SECRET=<REPLACE-WITH-A-RANDOM-SECRET>

[Install]
WantedBy=multi-user.target
```

Generate a random secret:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable bft-api
sudo systemctl start bft-api
sudo systemctl status bft-api
```

## 11. Cron Jobs (optional, for live data refresh)

```bash
crontab -e
```

```cron
# Populate latest technical data from CSVs nightly at 6:00 AM ET (11:00 UTC)
0 11 * * 1-5 cd /home/ubuntu/bar_fly_trading && BFT_DATA_DIR=/var/www/bft/data python -m webapp.backend.refresh_technicals >> /var/log/bft/technicals.log 2>&1

# Batch quote updates every 30 min during market hours (9:30-4:00 ET = 14:30-21:00 UTC)
*/30 14-20 * * 1-5 cd /home/ubuntu/bar_fly_trading && BFT_DATA_DIR=/var/www/bft/data python -m webapp.backend.update_quotes >> /var/log/bft/quotes.log 2>&1
```

## 12. Verify

- [ ] Visit `https://barflytrading.com` — should see the login page
- [ ] Register with invite code `BETA2026` — should redirect to sector dashboard
- [ ] Click a sector — should see stock grid
- [ ] Click a stock — should see the detail card
- [ ] Sign out — should return to login page
- [ ] Check `https://barflytrading.com/api/health` — should return `{"status": "ok"}`

---

## Quick Deploy Script (after first setup)

For future deploys, run on the EC2 server:

```bash
cd ~/bar_fly_trading
git pull origin feature/website
cd webapp/frontend && npm run build
cp -r dist/* /var/www/bft/frontend/
sudo systemctl restart bft-api
```

---

## Quick Reference

| What | Where |
|------|-------|
| Repo | `~/bar_fly_trading` |
| Frontend build | `/var/www/bft/frontend/` |
| JSON data files | `/var/www/bft/data/` |
| Auth database | `/var/www/bft/data/bft_auth.db` |
| FastAPI service | `sudo systemctl status bft-api` |
| Nginx config | `/etc/nginx/sites-available/bft` |
| SSL certs | Auto-renewed by certbot |
| Logs | `/var/log/bft/` |
| Invite code CLI | `BFT_DATA_DIR=/var/www/bft/data python -m webapp.backend.database CODE 10` |
