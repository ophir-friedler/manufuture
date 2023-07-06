import logging


# Validate that every project has at most 1 quote assigned to it
def validate_quote_to_single_project(all_tables_df):
    wp_type_quote = all_tables_df['wp_type_quote']
    if max(wp_type_quote.groupby(['project'])[['post_id']].count()['post_id']) > 1:
        logging.error("some project has more than 1 quote")
