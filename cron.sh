#!/bin/bash

# Cron script to pull new data and run the screener with PDF output.
# It's meant to be run on the EC2 instance, so if you want to run it on your local machine, you'll need to adjust the paths.

# If the tmp/cron.lock exists, then the script is already running - just abort
if [ -e /tmp/cron.lock ]; then
    echo "Script already running"
    exit 1
fi

# Create a lock file
touch /tmp/cron.lock
# Delete the lock file when the script finishes
trap "{ rm -f /tmp/cron.lock; }" EXIT SIGINT SIGTERM

cd /home/abettigole/bar_fly_trading
/usr/bin/python3 /home/abettigole/bar_fly_trading/api_data/pull_api_data.py -w all
/usr/bin/python3 /home/abettigole/bar_fly_trading/visualizations/screener.py --n_days 60 --data /home/abettigole/bar_fly_trading/all_data_0.csv --skip_sectors
/usr/bin/python3 /home/abettigole/bar_fly_trading/visualizations/pdf_overnight.py