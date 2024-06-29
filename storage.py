import os

from sqlalchemy import create_engine, MetaData, text

username = 'root'
password = os.getenv('MYSQL_PASSWORD', None)
host = '127.0.0.1'
port = 3306
dbname = 'bar_fly_trading'

if not password:
    raise Exception('Must set MYSQL_PASSWORD environment variable')


def create_database():
    return create_engine(connection_string)


def store_data(df, table_name):
    df.to_sql(table_name, engine, if_exists='replace', index_label='date')


def test_connection():
    metadata = MetaData()
    metadata.reflect(bind=engine)

    # Example custom query
    custom_query = """
    SELECT * from test;
    """

    # Execute the query
    with engine.connect() as connection:
        result = connection.execute(text(custom_query))

        # Fetch and print results
        for row in result:
            print(row)


connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}'
engine = create_database()