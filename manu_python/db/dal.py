# # Initialization and DB connection
import os

import mysql.connector
import pandas as pd
from mysql.connector import Error
from sqlalchemy import create_engine

from manu_python.config.config import MYSQL_PW, MYSQL_ROOT, MYSQL_MANUFUTURE_DB, MYSQL_HOST

RELEVANT_FOR_SERVING_TABLES_LIST = ['wp_type_manufacturer', 'wp_type_part', 'wp_type_project', 'wp_type_quote', 'pam_project_active_manufacturer_th_4_label_reqs']

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 300)
DB_CONNECTION_STRING = "mysql+pymysql://root:mysql123@localhost/manufuture"
EMAIL_LOGS_DIR = '/Users/ofriedler/Dropbox/Work/Consultation/Manufuture/dev/logs-2021-05-01_2021-12-30.csv'


# read a mysql table into a dataframe without using an existing db connection
def read_mysql_table_into_dataframe_without_connection(table_name) -> pd.DataFrame:
    sql_engine = create_engine(DB_CONNECTION_STRING)  # , pool_recycle=3600
    db_connection = sql_engine.connect()
    return pd.read_sql(f'SELECT * FROM ' + table_name, db_connection)

def mysql_table_to_dataframe(table_name, db_connection) -> pd.DataFrame:
    return pd.read_sql(f'SELECT * FROM ' + table_name, db_connection)


def get_db_connection():
    sql_engine = create_engine(DB_CONNECTION_STRING)  # , pool_recycle=3600
    db_connection = sql_engine.connect()
    return db_connection


# Load MySQL DB to all_tables_df
def fetch_all_tables_df():
    db_connection = get_db_connection()
    all_table_names = pd.read_sql(f'SHOW TABLES', db_connection)['Tables_in_manufuture']
    all_tables_df = {}
    for table in all_table_names:
        all_tables_df[table] = mysql_table_to_dataframe(table, db_connection)

    # Load e-mail logs to all_tables_df['email_logs']:
    # all_tables_df['email_logs'] = pd.read_csv(EMAIL_LOGS_DIR)
    return all_tables_df


# save all_tables_df to static data directory
def save_relevant_all_tables_df(all_tables_df):
    for table_name, table_df in all_tables_df.items():
        if table_name in RELEVANT_FOR_SERVING_TABLES_LIST:
            # table_df.to_csv('./manu_python/static_data/all_tables_df__' + table_name + '.csv', index=False)
            table_df.to_parquet('./manu_python/static_data/all_tables_df__' + table_name + '.parquet', index=False)


# read all_tables_df from static data directory
def load_relevant_all_tables_df():
    # Read all csvs from static data directory starting with 'all_tables_df__' and ending with '.csv' and load them to all_tables_df
    all_tables_df = {}
    for file in os.listdir('./manu_python/static_data/'):
        if file.startswith('all_tables_df__') and file.endswith('.parquet'):
            table_name = file.replace('all_tables_df__', '').replace('.parquet', '')
            if table_name in RELEVANT_FOR_SERVING_TABLES_LIST:
                all_tables_df[table_name] = pd.read_parquet('./manu_python/static_data/' + file)
            # pd.read_csv('./manu_python/static_data/' + file)
    return all_tables_df


def get_all_table_names(db_connection):
    return pd.read_sql(f'SHOW TABLES', db_connection)['Tables_in_manufuture']


# Create a table in MySQL DB
def create_table(table_name):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            print("Connected to MySQL database")

        cursor = connection.cursor()
        try:
            cursor.execute(f"CREATE TABLE {table_name} (id INT AUTO_INCREMENT PRIMARY KEY)")

            print(f"Table {table_name} created successfully")
        except Error as e:
            print(f"Error (create_table): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


# Drop table from mysql db manufuture
def drop_table(table_name):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            print("Connected to MySQL database")

        cursor = connection.cursor()
        try:
            cursor.execute(f"DROP TABLE {table_name}")
            print(f"Table {table_name} dropped successfully")
        except Error as e:
            print(f"Error (drop_table): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


# Add columns to a table in manufuture db
def add_columns_to_table(table_name, dict_columns_name_to_type):
    for column_name, column_type in dict_columns_name_to_type.items():
        add_column_to_table(table_name, column_name, column_type)


# Add column to a table in manufuture db
def add_column_to_table(table_name, column_name, column_type, verbose=False):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            if verbose:
                print("Connected to MySQL database")

        cursor = connection.cursor()
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD {column_name} {column_type}")
            if verbose:
                print(f"Column {column_name} added successfully to {table_name}")
        except Error as e:
            print(f"Error (add_column_to_table): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


# Insert a value to a column in a table in manufuture db
def insert_value_to_column(table_name, column_name, value):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            print("Connected to MySQL database")

        cursor = connection.cursor()
        try:
            cursor.execute(f"INSERT INTO {table_name} ({column_name}) VALUES ({value})")
            print(f"Value {value} inserted successfully to {table_name}.{column_name}")
        except Error as e:
            print(f"Error (insert_value_to_column): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


# Insert a row to a table in manufuture db
def insert_row_to_table(table_name, dict_columns_name_to_value):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            print("Connected to MySQL database")

        cursor = connection.cursor()
        try:
            query = generate_insert_query(dict_columns_name_to_value, table_name)
            cursor.execute(query)
        except Error as e:
            print("Query: " + f"INSERT INTO {table_name} ({columns_names}) VALUES ({values})")
            print(f"Error (insert_row_to_table): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


def generate_insert_query(dict_columns_name_to_value, table_name):
    prepare_sql_row_insert_syntax(dict_columns_name_to_value)
    columns_names = ",".join(dict_columns_name_to_value.keys())
    values = ",".join([str(value) for value in dict_columns_name_to_value.values()])
    query = f"INSERT INTO {table_name} ({columns_names}) VALUES ({values})"
    return query


# Get a dataframe, and build a mysql table from it according to the dataframe's name and columns
def dataframe_to_mysql_table(table_name, df):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            print("Connected to MySQL database")

        cursor = connection.cursor()
        try:
            sql_engine = create_engine(DB_CONNECTION_STRING)
            df.to_sql(table_name, con=sql_engine, if_exists='replace', index=False)
            print(f"Table {table_name} created successfully")
        except Error as e:
            print(f"Error (dataframe_to_mysql_table): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


def prepare_sql_row_insert_syntax(dict_columns_name_to_value):
    for key, value in dict_columns_name_to_value.items():
        if value is None:
            dict_columns_name_to_value[key] = 'NULL'
        if value == 'None':
            dict_columns_name_to_value[key] = 'NULL'
        if isinstance(value, str):
            dict_columns_name_to_value[key] = f"'{value}'"


def run_query_in_manufuture_db(query, verbose=False):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            if verbose:
                print("Connected to MySQL database")

        cursor = connection.cursor()
        try:
            cursor.execute(query)
            if verbose:
                print(query)
                print(f"Query {query} executed successfully")
        except Error as e:
            print(f"Error (run_query_in_manufuture_db): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


def run_queries_in_manufuture_db(queries, verbose=False):
    try:
        connection = mysql.connector.connect(host=MYSQL_HOST,
                                             database=MYSQL_MANUFUTURE_DB,
                                             user=MYSQL_ROOT,
                                             password=MYSQL_PW
                                             )
        if connection.is_connected():
            if verbose:
                print("Connected to MySQL database")

        cursor = connection.cursor()
        for query in queries:
            try:
                cursor.execute(query)
                if verbose:
                    print(query)
                    print(f"Query {query} executed successfully")
            except Error as e:
                print(f"Error (run_queries_in_manufuture_db): {e}, query: {query}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

