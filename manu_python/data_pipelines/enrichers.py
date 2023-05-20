import pandas as pd

from data_pipelines.table_builder import user_to_entity_rel
from utils import util_functions

import logging


# Enrich wp_quotes:
# Dependencies: all_tables_df['bids']
#
# competing_manufacturers: list of competing manufacturers
# num_candidates: number of competing manufacturers
# is_bid_chosen: boolean - is a bid selected by the customer
# competing_manufacturers: Competing manufacturers
# winning_manufacturers: Winning manufacturers
from utils.util_functions import parse_list_of_integers


def enrich_wp_quotes(all_tables_df):
    logging.info("Enriching wp_quotes with: competing_manufacturers, num_candidates, is_bid_chosen ")

    def get_bids_from_wp_quotes_row(row):
        bids = row['bids']
        if len(bids.strip()) == 0:
            return []
        if bids.isdigit():
            return [int(bids)]
        bids_split = bids[1:-1].split(",")
        return [int(bid) for bid in bids_split]

    def get_chosen_bids_from_wp_quotes_row(row):
        chosen_bids = row['chosen_bids']
        if (chosen_bids is None) or (len(chosen_bids.strip()) == 0):
            return []
        if chosen_bids.isdigit():
            return [int(chosen_bids)]
        bids_split = chosen_bids[1:-1].split(",")
        return [int(bid) for bid in bids_split]

    def get_the_bidding_manufacturer(bid_post_id):
        if bid_post_id in all_tables_df['bids'].index:
            return all_tables_df['bids'].loc[bid_post_id]['manufacturer_id']
        else:
            return None

    def competing_manufacturers(wp_quotes_row):
        bids = get_bids_from_wp_quotes_row(wp_quotes_row)
        ret_list = [get_the_bidding_manufacturer(bid_post_id) for bid_post_id in bids]
        ret_list = [x for x in ret_list if x is not None and len(x) > 0]
        ret_set = set(ret_list)
        if len(ret_set) != len(ret_list):
            logging.info("manufacturer bid multiple times: " + str(ret_list) + str(wp_quotes_row))
        return [x for x in ret_list if x is not None and len(x) > 0]

    def winning_manufacturers(wp_quotes_row):
        chosen_bids = get_chosen_bids_from_wp_quotes_row(wp_quotes_row)
        ret_val = [get_the_bidding_manufacturer(bid_post_id) for bid_post_id in chosen_bids]
        return [x for x in ret_val if x is not None and len(x) > 0]

    df = all_tables_df['wp_quotes']
    df['competing_manufacturers'] = df.apply(lambda row: competing_manufacturers(row), axis='columns')
    df['winning_manufacturers'] = df.apply(lambda row: winning_manufacturers(row), axis='columns')
    df['num_candidates'] = df['competing_manufacturers'].apply(len)
    df['is_bid_chosen'] = all_tables_df['wp_quotes']['chosen_bids'].isnull() == False
    # Translate project id to int:
    df['project'] = df['project'].astype('int64')
    all_tables_df['wp_quotes'] = df


# Dependencies: Enriched wp_quotes, all_tables_df['wp_posts'], all_tables_df['user_to_entity_rel']
# participation_count: number of times a manufacturer competed in a quote
# manufacturer_creation_date: Manufacturer creation date
# manufacturer_name
# vendor_status: vendor/pending_vendor
def enrich_wp_manufacturers(all_tables_df):
    # Enrich with participation_count
    logging.info("Enriching wp_manufacturers with: participation_count")

    # manufacturer_id -> participation count
    participation_count = {}
    for index, row in all_tables_df['wp_quotes'].iterrows():
        for manufacturer in row['competing_manufacturers']:
            if len(manufacturer) > 0:
                participation_count[manufacturer] = participation_count.get(manufacturer, 0) + 1

    participation_count_df = pd.DataFrame(participation_count.items(),
                                          columns=['manufacturer_id', 'participation_count']).astype('int64')
    # for key, value in participation_count.items():
    #     participation_count_df = participation_count_df.append(
    #         {'manufacturer_id': int(key), 'participation_count': int(value)}, ignore_index=True)

    df = pd.merge(all_tables_df['wp_manufacturers'], participation_count_df, how='left', left_on='post_id',
                  right_on='manufacturer_id').drop(columns=['manufacturer_id'])
    df['participation_count'] = df['participation_count'].fillna(0)
    all_tables_df['wp_manufacturers'] = df

    # Enrich with manufacturer creation date
    logging.info("Enriching wp_manufacturers with: manufacturer_creation_date")
    df = all_tables_df['wp_posts']
    wp_posts_manufacturers_df = df[df['post_type'] == 'manufacturer'][['ID', 'post_date', 'post_title']]
    wp_posts_manufacturers_df.columns = ['manufacturer_id', 'manufacturer_creation_date', 'manufacturer_name']

    df = pd.merge(all_tables_df['wp_manufacturers'], wp_posts_manufacturers_df, how='left', left_on='post_id',
                  right_on='manufacturer_id').drop(columns=['manufacturer_id'])

    # Parse creation date year month day
    df = util_functions.add_columns_year_month_day_from_datetime(df,
                                                            columns_prefix='manufacturer_creation_date',
                                                            datetime_column='manufacturer_creation_date')

    all_tables_df['wp_manufacturers'] = df

    user_to_entity_rel = all_tables_df['user_to_entity_rel']
    user_to_manuf_status = user_to_entity_rel[user_to_entity_rel['user_type'] == 'manufacturer'][['user_type_post_id', 'user_type_status']]
    def has_vendor(group):
        user_type_status_set = set(group['user_type_status'].unique())
        if 'vendor' in user_type_status_set:
            return pd.Series(['vendor']) # pd.DataFrame([])
        if 'pending_vendor' in user_type_status_set:
            return pd.Series([ 'pending_vendor']) # pd.DataFrame([])

    manuf_to_status = user_to_manuf_status.groupby('user_type_post_id').apply(lambda group: has_vendor(group)).reset_index()
    manuf_to_status.columns = ['post_id', 'vendor_status']

    df = pd.merge(all_tables_df['wp_manufacturers'], manuf_to_status, how='left', left_on='post_id',
                  right_on='post_id')

    all_tables_df['wp_manufacturers'] = df


# # First manufacturer bid:
# def first_month_manufacturer_bid(manufacturer_id):
#     df = all_tables_df['bids']
#     manufacturer_bids_df = df[df['manufacturer_id'].apply(str) == str(manufacturer_id)]


# Enrich bids dataframe with 'is_bid_chosen'
# Dependencies: wp_quotes
def add_is_bid_chosen_to_bids_df(all_tables_df):
    # list of post ids of chosen bids
    all_chosen_bid_ids = []
    df = all_tables_df['wp_quotes']
    df[df['is_bid_chosen']]['chosen_bids'].apply(
        lambda bids: util_functions.extend_list_of_bids(all_chosen_bid_ids, bids))
    all_chosen_bid_ids = list(map(int, all_chosen_bid_ids))
    all_tables_df['bids']['is_bid_chosen'] = all_tables_df['bids'].apply(
        lambda row: 1 if row.name in all_chosen_bid_ids else 0, axis='columns')


# Dependencies: all_tables_df['wp_posts'], all_tables_df['wp_parts']
# Add project creation date to wp_projects
# Add num days from creation to approval
# Parse project_creation_date year month day
# Parse approval_date year month day
# Add is_quote_carried_out (1 if it has a wp_quote entry, 0 otherwise)
def enrich_wp_projects(all_tables_df):
    # Typing
    df = all_tables_df['wp_projects']
    df['post_id'] = df['post_id'].astype('int64')
    df['approval_date'] = pd.to_datetime(df['approval_date'], format='%Y%m%d', errors='coerce')
    df['req_milling'] = df['req_milling'].fillna(0).astype('int64')
    df['req_turning'] = df['req_turning'].fillna(0).astype('int64')
    df['one_manufacturer'] = df['one_manufacturer'].fillna(0).astype('int64')
    all_tables_df['wp_projects'] = df

    # Add project creation date to wp_projects
    wp_projects_add_creation_date(all_tables_df)

    df = all_tables_df['wp_projects']

    # Remove projects that do not appear in wp_posts (ad-hoc)
    df = df[df['project_creation_date'].isnull() == False]

    # Parse project_creation_date year month day
    df = util_functions.add_columns_year_month_day_from_datetime(df,
                                                            columns_prefix='project_creation_date',
                                                            datetime_column='project_creation_date')

    # Parse approval_date year month day
    df = util_functions.add_columns_year_month_day_from_datetime(df,
                                                            columns_prefix='approval_date',
                                                            datetime_column='approval_date')

    # Add num days from creation to approval
    df['num_days_from_creation_to_approval'] = df.apply(
        lambda row: util_functions.diff_days_or_none(start_date=row['project_creation_date'],
                                                     end_date=row['approval_date']), axis=1)

    # Add number of distinct parts
    df['num_distinct_parts'] = df.apply(lambda row: len(row['parts'].split(",")) if row['parts'] is not None else 0, axis=1)

    # Bin number of distinct parts
    df['num_distinct_parts_binned'] = df.apply(lambda row: bin_feature(row['num_distinct_parts'], [1, 4, 11, 20]), axis=1)

    # Add total number of parts
    # Create a dictionary to map part IDs to their quantities
    parts_dict = all_tables_df['wp_parts'].set_index('post_id')['quantity'].astype(int).to_dict()

    # Function to calculate total quantity of parts for a project
    def calculate_total_quantity_of_parts(row, parts_dict):
        total_quantity_of_parts = 0
        for part_id in parse_list_of_integers(row['parts']):
            quantity = int(parts_dict.get(part_id, 0))
            total_quantity_of_parts += quantity
        return total_quantity_of_parts

    # Apply the function to the 'parts' column of the projects dataframe
    df['total_quantity_of_parts'] = df.apply(calculate_total_quantity_of_parts, axis=1, args=(parts_dict,))

    # bin total_quantity_of_parts
    df['total_quantity_of_parts_binned'] = df.apply(lambda row: bin_feature(row['total_quantity_of_parts'], [0, 1, 10, 30, 100, 200]), axis=1)

    all_tables_df['wp_projects'] = df

    # Add is_quote_carried_out (1 if it has a wp_quote entry, 0 otherwise)
    df = all_tables_df['wp_quotes'][['project', 'post_id']].rename(columns={'post_id': 'is_quote_carried_out'})
    is_quote_carried_out_for_project_df = df.groupby(['project']).count().reset_index()
    df = pd.merge(all_tables_df['wp_projects'], is_quote_carried_out_for_project_df, how='left', left_on='post_id',
                  right_on='project').drop(columns=['project'])  # .rename(columns={'post_id': 'is_quote_carried_out'})
    df['is_quote_carried_out'] = df['is_quote_carried_out'].fillna(0).astype('int64')
    all_tables_df['wp_projects'] = df


def bin_feature(feature_value, bins_arr):
    bins_arr.sort()
    if feature_value < bins_arr[0]:
        return  "<" + str(bins_arr[0])
    for idx, bin_upper_bound in enumerate(bins_arr) :
        if feature_value < bin_upper_bound:
            return "[" + str(bins_arr[idx-1]) + "-" + str(bins_arr[idx]) + ")"
    if feature_value >= bins_arr[-1]:
        return ">=" + str(bins_arr[-1])


def wp_projects_add_creation_date(all_tables_df):
    wp_posts = all_tables_df['wp_posts']
    project_to_creation_date_df = wp_posts[wp_posts['post_type'] == 'project'][['ID', 'post_date']]
    df = pd.merge(all_tables_df['wp_projects'], project_to_creation_date_df, how='left', left_on='post_id',
                  right_on='ID').drop(columns=['ID']).rename(columns={'post_date': 'project_creation_date'})
    all_tables_df['wp_projects'] = df


def enrich_all(all_tables_df):
    enrich_wp_quotes(all_tables_df)
    add_is_bid_chosen_to_bids_df(all_tables_df)
    enrich_wp_manufacturers(all_tables_df)
    enrich_wp_projects(all_tables_df)

