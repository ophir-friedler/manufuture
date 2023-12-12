import logging

from manu_python.config.config import WERK_START_DIR
from manu_python.data_pipelines import table_builder, enrichers, aggregators, validators
from manu_python.db.dal import fetch_all_tables_df
from manu_python.utils import util_functions


def get_all_tables_df(logging_level=logging.WARNING, from_scratch=True):
    set_logging_level_to_info(logging_level=logging_level)
    # Execute data pipelines
    if from_scratch:
        table_builder.build_werk_from_starting_directory(WERK_START_DIR)
    all_tables_df = fetch_all_tables_df()
    table_builder.build_raw_data_tables(all_tables_df, from_scratch=from_scratch)

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



