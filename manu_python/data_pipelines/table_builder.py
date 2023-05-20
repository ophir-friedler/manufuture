import logging

import pandas as pd

from config.config import COUNTRY_TO_ISO_MAP
from utils import util_functions


# Builds all_tables_df['bids'] dataframe:
# Dependencies: all_tables_df['wp_postmeta'], all_tables_df['wp_posts']
def bids(all_tables_df):
    all_bid_ids = util_functions.get_all_bid_ids(all_tables_df)
    bids_df = pd.DataFrame(data={'post_id': list(all_bid_ids)})

    # Collect data from wp_postmeta
    df = all_tables_df['wp_postmeta'][['post_id', 'meta_key', 'meta_value']]
    df = df[df['post_id'].isin(all_bid_ids)]

    bids_in_postmeta_df = df[df['meta_key'] == 'manufacturer'] \
        .rename(columns={'post_id': 'post_id', 'meta_value': 'manufacturer_id'}) \
        .drop(columns=['meta_key']) \
        .reset_index(drop=True)

    all_tables_df['bids'] = pd.merge(bids_df, bids_in_postmeta_df, on="post_id", how="left").fillna('') \
        .rename(columns={'post_id': 'bid_post_id'}) \
        .set_index('bid_post_id')

    # Collect data from wp_posts
    df = all_tables_df['wp_posts'][['ID', 'post_date', 'post_type']]
    df = df[df['ID'].isin(all_bid_ids)]
    df = pd.merge(all_tables_df['bids'], df, left_index=True, right_on="ID", how="left") \
        .rename(columns={'ID': 'bid_post_id'}) \
        .set_index('bid_post_id')

    # Clean out bids that don't have data (lost/ erased)
    df = df[df['post_type'].isnull() == False]

    # Parse year month day
    df = util_functions.add_columns_year_month_day_from_datetime(df,
                                                                 columns_prefix='post_date',
                                                                 datetime_column='post_date')

    all_tables_df['bids'] = df


# Dependencies: wp_quotes (enriched), wp_projects, wp_manufacturers
# Builds pm_project_manufacturer
def pm_project_manufacturer(all_tables_df):
    # training data + labels are based on wp_quotes
    # Get all project ids that have quotes
    wp_quotes = all_tables_df['wp_quotes'][
        ['post_id', 'bids', 'project', 'competing_manufacturers', 'winning_manufacturers']]
    # Join projects features
    wp_projects = all_tables_df['wp_projects']
    pm_df = wp_quotes.merge(wp_projects, left_on='project', right_on='post_id', suffixes=('_quote', '_project'))
    # For each manufacturer create a project-manufacturer row with data from wp_manufacturers
    wp_manufacturers = all_tables_df['wp_manufacturers'].rename(columns={'post_id': 'post_id_manuf'})
    pm_df = pm_df.merge(wp_manufacturers, how='cross', suffixes=('_quote', '_manuf'))

    # Clean columns data
    standardize_country_values(pm_df)

    # build Label column
    did_manufacturer_bid(training_data=pm_df, label_column='is_manuf_bid')
    # set index by project-manufacturer
    pm_df = pm_df.set_index(['post_id_project', 'post_id_manuf'])
    all_tables_df['pm_project_manufacturer'] = pm_df


# # Filter out non-participating manufacturers from pm_project_manufacturer
# # Dependencies: pm_project_manufacturer
# def pam_project_active_manufacturer(all_tables_df):
#     pam_df = all_tables_df['pm_project_manufacturer']
#     manuf_participation_df = pam_df.groupby('post_id_manuf')[['is_manuf_bid']].sum()
#     list_of_participating_manufs = list(manuf_participation_df[manuf_participation_df['is_manuf_bid'] != 0].index)
#     pam_df = pam_df[pam_df.index.get_level_values('post_id_manuf').isin(list_of_participating_manufs)]
#     all_tables_df['pam_project_active_manufacturer'] = pam_df


# Dependencies: pam_project_active_manufacturer
def ac_agency_manufacturer(all_tables_df):
    ac_df = all_tables_df['pm_project_manufacturer'] \
        .reset_index().groupby(['agency', 'post_id_manuf'])[['is_manuf_bid']] \
        .agg({'is_manuf_bid': ['sum']})
    ac_df.columns = ['num_bids']
    all_tables_df['ac_agency_manufacturer'] = ac_df


# Create pam_project_active_manufacturer_th_<num_bids_activation_threshold>
def pam_project_active_manufacturer(all_tables_df, num_bids_activation_threshold):
    new_table_name: str = 'pam_project_active_manufacturer_th_' + str(num_bids_activation_threshold)
    if new_table_name not in all_tables_df:
        pm = all_tables_df['pm_project_manufacturer'].reset_index()
        manuf_num_bids = pm.groupby(['post_id_manuf'])[['is_manuf_bid']].sum().reset_index()
        manuf_num_bids.columns = ['post_id_manuf', 'manuf_num_bids']
        active_manufs = manuf_num_bids[manuf_num_bids['manuf_num_bids'] >= num_bids_activation_threshold]
        pm_only_active_manufs = pd.merge(pm, active_manufs, on='post_id_manuf', how='inner')
        all_tables_df[new_table_name] = pm_only_active_manufs
        logging.info(new_table_name + "created in all_tables_df")

    return new_table_name


def pam_filter_by_project_requirements(all_tables_df, pam_table_name):
    new_table_name: str = pam_table_name + '_filter_reqs'
    # for efficiency
    if new_table_name not in all_tables_df:
        start_table = all_tables_df[pam_table_name]
        all_tables_df[new_table_name] = start_table[(start_table['cnc_milling'] >= start_table['req_milling'])
                                                    & (start_table['cnc_turning'] >= start_table['req_turning'])].copy()

    return new_table_name


def pam_label_by_project_requirements(all_tables_df, pam_table_name):
    new_table_name: str = pam_table_name + '_label_reqs'
    # for efficiency
    if new_table_name not in all_tables_df:
        start_table = all_tables_df[pam_table_name]
        start_table[(start_table['cnc_milling'] < start_table['req_milling'])
                    | (start_table['cnc_turning'] < start_table['req_turning'])]['is_manuf_bid'] = 0
        all_tables_df[new_table_name] = start_table.copy()

    return new_table_name


def did_manufacturer_bid(training_data, label_column):
    def label_is_manuf_bid(row):
        competing_manufs = list(map(int, row['competing_manufacturers']))
        current_manuf = row['post_id_manuf']
        return 1 if current_manuf in competing_manufs else 0

    training_data[label_column] = training_data.apply(lambda row: label_is_manuf_bid(row), axis='columns')


# Build pam + filter by project requirements
def build_proj_manu_training_table(all_tables_df, min_num_manufacturer_bids):
    pam_th_table_name = pam_project_active_manufacturer(all_tables_df, min_num_manufacturer_bids)
    # pam_th_req_filter_table_name = pam_filter_by_project_requirements(all_tables_df, pam_th_table_name)
    pam_th_req_label_table_name = pam_label_by_project_requirements(all_tables_df, pam_th_table_name)
    return pam_th_req_label_table_name


def standardize_country_values(training_data):
    if 'country' in training_data.columns:
        training_data['country'] = training_data['country'].transform(
            lambda val: logging.error(": value " + val + " not in country map") if val not in COUNTRY_TO_ISO_MAP.keys()
            else COUNTRY_TO_ISO_MAP[val])
    else:
        logging.warning('country not in training_data.columns')


# Return: 'user_id', 'user_type', 'user_type_post_id', 'user_type_status'
def extract_user_info_from_wp_usermeta(user_id_group):
    manufacturer_info = user_id_group[(user_id_group['meta_key'] == 'rel_manufacturer') & (user_id_group['meta_value'].str.len() > 0)]
    agency_info = user_id_group[(user_id_group['meta_key'] == 'rel_agency') & (user_id_group['meta_value'].str.len() > 0)]
    status_info = user_id_group[user_id_group['meta_key'] == 'wp_capabilities']
    manufacturer_name = manufacturer_info.iloc[0]['meta_value'] if len(manufacturer_info) > 0 else None
    agency_name = agency_info.iloc[0]['meta_value'] if len(agency_info) > 0 else None
    # status_detail = a:1:{s:14:"pending_vendor";b:1;}
    user_id = user_id_group.iloc[0]['user_id']
    result = []
    if manufacturer_name is not None:
        manufacturer_status = ""
        if len(status_info.iloc[0]['meta_value']) > 0 and "\"pending_vendor\"" in status_info.iloc[0]['meta_value']:
            manufacturer_status = 'pending_vendor'
        if len(status_info.iloc[0]['meta_value']) > 0 and "\"vendor\"" in status_info.iloc[0]['meta_value']:
            manufacturer_status = 'vendor'
        result.append((user_id, 'manufacturer', manufacturer_name, manufacturer_status))
    if agency_name is not None:
        result.append((user_id, 'agency', agency_name, None))
    return pd.DataFrame(result, columns=['user_id', 'user_type', 'user_type_post_id', 'user_type_status'])


# Dependencies: all_tables_df['wp_usermeta']
# user type (manufacturer, agency), post_id of user type, status (of manufacturer - vendor/pending_vendor)
def user_to_entity_rel(all_tables_df):
    logging.info("Building user_to_entity_rel: user_id, user_type, user_type_post_id, user_type_status")

    # group wp_usermeta by user_id and apply the extract_user_info_from_wp_usermeta function
    user_info_df = all_tables_df['wp_usermeta'].groupby('user_id').apply(extract_user_info_from_wp_usermeta).reset_index(drop=True)
    user_info_df['user_type_post_id'] = user_info_df['user_type_post_id'].fillna(-1).astype(int)
    all_tables_df['user_to_entity_rel'] = user_info_df


