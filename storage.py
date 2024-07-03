import os

from sqlalchemy import create_engine, MetaData, text

import pandas as pd

username = 'root'
password = os.getenv('MYSQL_PASSWORD', None)
host = '127.0.0.1'
port = 3306
dbname = 'bar_fly_trading'

if not password:
    raise Exception('Must set MYSQL_PASSWORD environment variable')


def create_database():
    return create_engine(connection_string)


def store_data(df, table_name, index=True):
    # TODO: Add an option to append instead of replace for incremental updates
    df.to_sql(table_name, engine, if_exists='replace', index=index)


def select_all_from_table(table_name: str, order_by: str, limit: int = 10):
    query = f"""
    SELECT * from {table_name} {f'ORDER BY {order_by} desc' if order_by else ''} LIMIT {limit};
    """
    df = pd.read_sql_query(query, engine)
    print(f'First {limit} rows from {table_name}:')
    pd.set_option('display.max_columns', None)
    print(df)


connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}'
engine = create_database()
