import logging

from manu_python.config.config import WERK_START_DIR
from manu_python.data_pipelines import table_builder, enrichers, aggregators, validators
from manu_python.data_pipelines.table_builder import werk_by_result_name
from manu_python.db.dal import fetch_all_tables_df, generate_insert_query, \
    run_queries_in_manufuture_db, dataframe_to_mysql_table
from manu_python.external_data_processing.werk_data_processing import process_all_werk_results_dirs_to_df, \
    build_werk_table
from manu_python.utils import util_functions


def get_all_tables_df(logging_level=logging.WARNING):
    set_logging_level_to_info(logging_level=logging_level)
    # Execute data pipelines
    build_werk_from_starting_directory(WERK_START_DIR)
    all_tables_df = fetch_all_tables_df()
    table_builder.build_raw_data_tables(all_tables_df)

    enrichers.enrich_all(all_tables_df)

    # Model training data
    table_builder.build_training_data_tables(all_tables_df)

    # Aggregated statistics
    aggregators.monthly_bid_success_rate_df(all_tables_df)
    aggregators.monthly_projects_stats(all_tables_df)
    aggregators.monthly_manufacturers_stats(all_tables_df)
    aggregators.stats_by_num_candidates(all_tables_df)

    validators.validate_quote_to_single_project(all_tables_df)

    # find all rows in wp_type_part that
    return all_tables_df


def get_all_bid_ids(all_tables_df):
    return util_functions.get_all_bid_ids(all_tables_df)


def set_logging_level_to_info(logging_level):
    logging.basicConfig(level=logging_level)


def build_werk_from_starting_directory(starting_dir):
    build_werk_table()
    all_results_list = process_all_werk_results_dirs_to_df(starting_dir)
    insert_statements = [generate_insert_query(dict_werk_column_name_to_value, 'werk') for dict_werk_column_name_to_value in all_results_list]
    run_queries_in_manufuture_db(insert_statements)
    werk_by_name_df = werk_by_result_name()
    dataframe_to_mysql_table('werk_by_name', werk_by_name_df)



