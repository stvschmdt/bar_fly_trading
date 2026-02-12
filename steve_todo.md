# Steve's Hosting Prep TODO

Step-by-step guide to get the BFT web app hosted on your domain with AWS.

---

## 1. Domain DNS Setup

- [ ] Log into your domain registrar (Route 53, Namecheap, etc.)
- [ ] Create an **A record** pointing your domain (e.g. `barflytrading.com`) to your EC2 public IP
- [ ] Create a **CNAME** for `www` → same domain
- [ ] Optional: create `api.barflytrading.com` A record → same EC2 IP (or use path-based routing `/api/`)
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

Run these on the EC2 instance:

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
pip install fastapi uvicorn[standard] yfinance aiofiles
```

## 4. SSL Certificate (Let's Encrypt)

```bash
# Get cert (replace with your actual domain)
sudo certbot --nginx -d barflytrading.com -d www.barflytrading.com

# Verify auto-renewal
sudo certbot renew --dry-run
```

## 5. Nginx Configuration

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

    # FastAPI backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket (Phase 2)
    location /ws/ {
        proxy_pass http://127.0.0.1:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/bft /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## 6. Deploy Directory Setup

```bash
sudo mkdir -p /var/www/bft/{frontend,data}
sudo chown -R $USER:$USER /var/www/bft
```

## 7. Systemd Service for FastAPI

Create `/etc/systemd/system/bft-api.service`:

```ini
[Unit]
Description=BFT FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/bar_fly_trading/webapp/backend
ExecStart=/home/ubuntu/miniforge3/bin/uvicorn api:app --host 127.0.0.1 --port 8000
Restart=always
Environment=BFT_DATA_DIR=/var/www/bft/data

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable bft-api
sudo systemctl start bft-api
```

## 8. Cron Jobs

```bash
crontab -e
```

Add:
```cron
# Daily report generation at 6:00 AM ET (11:00 UTC)
0 11 * * 1-5 cd /home/ubuntu/bar_fly_trading && python -m webapp.backend.generate_reports >> /var/log/bft/reports.log 2>&1

# Quote updates every 5 min during market hours (9:30-4:00 ET = 14:30-21:00 UTC)
*/5 14-20 * * 1-5 cd /home/ubuntu/bar_fly_trading && python -m webapp.backend.update_quotes >> /var/log/bft/quotes.log 2>&1
```

```bash
sudo mkdir -p /var/log/bft
sudo chown $USER:$USER /var/log/bft
```

## 9. Build & Deploy (first time)

```bash
# On your dev machine, build the React frontend
cd webapp/frontend
npm install
npm run build

# Copy build output to EC2
scp -r dist/* ubuntu@YOUR_EC2_IP:/var/www/bft/frontend/

# Or on EC2 directly:
cd ~/bar_fly_trading/webapp/frontend
npm install && npm run build
cp -r dist/* /var/www/bft/frontend/

# Generate initial data files
cd ~/bar_fly_trading
python -m webapp.backend.build_sector_map
python -m webapp.backend.generate_reports --symbols SPY,QQQ,AAPL,NVDA,JPM  # start small
```

## 10. Verify

- [ ] Visit `https://barflytrading.com` — should see the sector grid
- [ ] Click a sector — should see stock grid
- [ ] Click a stock — should see the detail card
- [ ] Check `https://barflytrading.com/api/sectors` — should return JSON

---

## Quick Reference

| What | Where |
|------|-------|
| Frontend build | `/var/www/bft/frontend/` |
| JSON data files | `/var/www/bft/data/` |
| FastAPI service | `systemctl status bft-api` |
| Nginx config | `/etc/nginx/sites-available/bft` |
| SSL certs | Auto-renewed by certbot |
| Logs | `/var/log/bft/` |
