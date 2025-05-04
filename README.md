# bar_fly_trading
Research &amp; Development on Finance Projects

## Local Setup
Python:
- Install Python & pip
- We don't have a requirements.txt yet, so if you get a `ModuleNotFoundError` while running, you may need to install the module with `pip install <module>`

Docker:
- Install Docker Desktop

MySQL: 
- Pull MySQL Docker Image
```
docker pull mysql:latest
```
- Run MySQL Docker Container and create a database
```
docker run --name mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -e MYSQL_DATABASE=bar_fly_trading -p 3306:3306 -d mysql:latest
```
- Connect to MySQL Docker Container, and enter password when prompted
```
mysql -h 127.0.0.1 -P 3306 -u root -p
```
- Set your MySQL password in `MYSQL_PASSWORD` environment variable so the script can access it.
```
export MYSQL_PASSWORD=my-secret-pw
```

## Connect to Remote DB
- You can open a read-only connection to the remote DB running on our EC2 instance using the `readonly_user`.
- Set your MySQL password in `MYSQL_READONLY_PASSWORD` environment variable so the script can access it. The password currently has an exclamation point in it, which must be escaped.
```
export MYSQL_READONLY_PASSWORD="my-secret-\!pw"
```
- Open an SSH tunnel, exposing the EC2's MySQL port `3306` to port `3307` on your local machine. Your local MySQL DB is using port `3306`, so we use `3307` to avoid conflicts.
```
ssh -L 3307:localhost:3306 username@54.90.246.184 -N -i <path_to_private_key>
```
- If running `pull_api_data.py` or `backtest.py`, set the `--db` flag to `remote`.
- If you just want to query the remote DB from a MySQL shell:
```
mysql -h 127.0.0.1 -P 3307 -u readonly_user -p
```
```

Alphavantage:
- Set your Alphavantage API key in `ALPHAVANTAGE_API_KEY` environment variable so the script can access it. 

RlLib:

## initialize (clean) db
Symbols are currently hardcoded into main.py, so put in the symbols you're interested in, and run `python main.py`. The script will fetch the desired data and stored it in the DB.

--steps to convert sql to pandas df (dates, symbols)

## advanced fetch
--steps to generate data for list of symbols

--convert sql to pandas df (dates, symbols)

## simple analyze (todo)
--steps to run visual analysis / verification module (dates, symbol)

