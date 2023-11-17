import logging

from manu_python.config.config import WERK_START_DIR
from manu_python.data_pipelines import table_builder, enrichers, aggregators, validators
from manu_python.data_pipelines.table_builder import werk_by_result_name
from manu_python.db.dal import fetch_all_tables_df, generate_insert_query, \
    run_queries_in_manufuture_db, dataframe_to_mysql_table
from manu_python.external_data_processing.werk_data_processing import process_all_werk_results_dirs_to_df, \
    build_werk_table
from manu_python.utils import data_cleaning, util_functions

LABEL_COLUMN_NAME = 'is_manuf_bid'


def get_all_tables_df():
    # Execute data pipelines
    build_werk_from_starting_directory(WERK_START_DIR)
    all_tables_df = fetch_all_tables_df()
    table_builder.get_wp_tables_by_post_type(all_tables_df)
    data_cleaning.clean_tables(all_tables_df)
    table_builder.user_to_entity_rel(all_tables_df)
    table_builder.netsuite_prices(all_tables_df)

    enrichers.enrich_all(all_tables_df)

    table_builder.pm_project_manufacturer(all_tables_df)
    table_builder.pam_project_active_manufacturer(all_tables_df, 1)

    # Aggregated statistics
    aggregators.monthly_bid_success_rate_df(all_tables_df)
    aggregators.monthly_projects_stats(all_tables_df)
    aggregators.monthly_manufacturers_stats(all_tables_df)
    aggregators.stats_by_num_candidates(all_tables_df)
    table_builder.ac_agency_manufacturer(all_tables_df)

    validators.validate_quote_to_single_project(all_tables_df)

    # find all rows in wp_type_part that
    return all_tables_df


def get_all_bid_ids(all_tables_df):
    return util_functions.get_all_bid_ids(all_tables_df)


def set_logging_level_to_info():
    logging.basicConfig(level=logging.INFO)


def build_werk_from_starting_directory(starting_dir):
    build_werk_table()
    all_results_list = process_all_werk_results_dirs_to_df(starting_dir)
    insert_statements = [generate_insert_query(dict_werk_column_name_to_value, 'werk') for dict_werk_column_name_to_value in all_results_list]
    run_queries_in_manufuture_db(insert_statements)
    werk_by_name_df = werk_by_result_name()
    dataframe_to_mysql_table('werk_by_name', werk_by_name_df)



