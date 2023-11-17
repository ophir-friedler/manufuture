# # Initialization and DB connection

import mysql.connector
import pandas as pd
from mysql.connector import Error
from sqlalchemy import create_engine

from manu_python.config.config import MYSQL_PW, MYSQL_ROOT, MYSQL_MANUFUTURE_DB, MYSQL_HOST

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 300)
DB_CONNECTION_STRING = "mysql+pymysql://root:mysql123@localhost/manufuture"
EMAIL_LOGS_DIR = '/Users/ofriedler/Dropbox/Work/Consultation/Manufuture/dev/logs-2021-05-01_2021-12-30.csv'


# read a mysql table into a dataframe without using an existing db connection
def mysql_table_to_dataframe_without_connection(table_name) -> pd.DataFrame:
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
            print(f"Error: {e}")
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
            print(f"Error: {e}")
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
            print(f"Error: {e}")
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
            print(f"Error: {e}")
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
            print(f"Error: {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")


def generate_insert_query(dict_columns_name_to_value, table_name):
    prepare_sql_row_insert_syntax(dict_columns_name_to_value)
    columns_names = ",".join(dict_columns_name_to_value.keys())
    values = ",".join(dict_columns_name_to_value.values())
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
            print(f"Error: {e}")
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
            print(f"Error: {e}")
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
        try:
            for query in queries:
                cursor.execute(query)
                if verbose:
                    print(query)
                    print(f"Query {query} executed successfully")
        except Error as e:
            print(f"Error: {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
