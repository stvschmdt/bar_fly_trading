# bar_fly_trading
Research &amp; Development on Finance Projects

## setup
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
EXPORT MYSQL_ROOT_PASSWORD=my-secret-pw
EXPORT MYSQL_PASSWORD=my-secret-pw
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

