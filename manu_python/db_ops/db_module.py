# # Initialization and DB connection

import pandas as pd

from sqlalchemy import create_engine

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 300)
DB_CONNECTION_STRING = "mysql+pymysql://root:mysql123@localhost/manufuture"
EMAIL_LOGS_DIR = '/Users/ofriedler/Dropbox/Work/Consultation/Manufuture/dev/logs-2021-05-01_2021-12-30.csv'


def selectAll(table_name, db_connection):
    return pd.read_sql(f'SELECT * FROM ' + table_name, db_connection)


def get_db_connection():
    sql_engine = create_engine(DB_CONNECTION_STRING)  # , pool_recycle=3600
    db_connection = sql_engine.connect()
    return db_connection


# Load MySQL DB to all_tables_df
def build_all_tables_df():
    db_connection = get_db_connection()
    all_table_names = pd.read_sql(f'SHOW TABLES', db_connection)['Tables_in_manufuture']
    all_tables_df = {}
    for table in all_table_names:
        all_tables_df[table] = selectAll(table, db_connection)

    # Load e-mail logs to all_tables_df['email_logs']:
    # all_tables_df['email_logs'] = pd.read_csv(EMAIL_LOGS_DIR)
    return all_tables_df


def get_all_table_names(db_connection):
    return pd.read_sql(f'SHOW TABLES', db_connection)['Tables_in_manufuture']
