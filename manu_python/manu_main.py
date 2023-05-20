import logging

from data_pipelines import table_builder, enrichers, aggregators, validators
from db_ops.db_module import build_all_tables_df
from utils import data_cleaning, util_functions

LABEL_COLUMN_NAME = 'is_manuf_bid'


def get_all_tables_df():
    # Execute data pipelines
    all_tables_df = build_all_tables_df()
    data_cleaning.clean_tables(all_tables_df)
    table_builder.bids(all_tables_df)
    table_builder.get_wp_tables_by_post_type(all_tables_df)
    table_builder.user_to_entity_rel(all_tables_df)

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
    return all_tables_df


def get_all_bid_ids(all_tables_df):
    return util_functions.get_all_bid_ids(all_tables_df)


def set_logging_level(level):
    logging.basicConfig(level=level)


