import os
from enum import Enum

from sqlalchemy import create_engine, text

import pandas as pd

username = 'root'
password = os.getenv('MYSQL_PASSWORD', None)
host = '127.0.0.1'
port = 3306
dbname = 'bar_fly_trading'


# Methods for writing to a table. These methods apply at a table level, not a row level.
# i.e. APPEND will add new rows to the table, REPLACE will replace the entire table with the new data.
class TableWriteOption(Enum):
    APPEND = 'append'
    REPLACE = 'replace'


if not password:
    raise Exception('Must set MYSQL_PASSWORD environment variable')


def create_database():
    return create_engine(connection_string)


def store_data(df, table_name, write_option: TableWriteOption, include_index=True):
    if df.empty:
        print(f'No data to store in {table_name}.')
        return

    # TODO: When replacing a table, we should drop and recreate the table manually so we can create PKs.
    # TODO: This will guard against bugs resulting in duplicate data (e.g. core_stock for same date-symbol combo)
    df.to_sql(table_name, engine, if_exists=write_option.value, index=include_index)


def select_all_from_table(table_name: str, order_by: str, limit: int = 10):
    query = f"""
    SELECT * from {table_name} {f'ORDER BY {order_by} desc' if order_by else ''} LIMIT {limit};
    """
    df = pd.read_sql_query(query, engine)
    print(f'First {limit} rows from {table_name}:')
    pd.set_option('display.max_columns', None)
    print(df)


def get_last_updated_date(table_name: str, date_col: str, symbol: str):
    # Get the most recent date in the table with an optional symbol filter (not all tables have symbols)
    query = text(f"SELECT MAX({date_col}) FROM {table_name}{' WHERE symbol = :symbol' if symbol else ''};")
    with engine.connect() as connection:
        result = connection.execute(query, {"symbol": symbol})
        last_updated_date = pd.to_datetime(result.fetchone()[0])
    return last_updated_date


connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}'
engine = create_database()
