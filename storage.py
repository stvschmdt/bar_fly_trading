# database.py

from sqlalchemy import create_engine

def create_database(db_name='historical_data.db'):
    engine = create_engine(f'sqlite:///{db_name}')
    return engine

def store_data(df, engine, table_name):
    df.to_sql(table_name, engine, if_exists='replace', index_label='date')

# Initialize the database
engine = create_database()

