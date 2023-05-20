import logging


# Validate taht every project has at most 1 quote assigned to it
def validate_quote_to_single_project(all_tables_df):
    wp_quotes = all_tables_df['wp_quotes']
    if max(wp_quotes.groupby(['project'])[['post_id']].count()['post_id']) > 1:
        logging.warning("some project has more than 1 quote")

