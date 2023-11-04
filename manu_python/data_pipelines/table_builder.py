import logging

import pandas as pd

from manu_python.config.config import COUNTRY_TO_ISO_MAP


def netsuite_prices(all_tables_df):
    netsuite_prices_df = pd.read_csv('/Users/ofriedler/Dropbox/Work/Consultation/Manufuture/dev/manufuture/netsuite_prices.csv')
    netsuite_prices_df['average_Rate'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('mean')
    netsuite_prices_df['average_Rate'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('mean')
    netsuite_prices_df['min_Rate'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('min')
    netsuite_prices_df['max_Rate'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('max')
    netsuite_prices_df['Currencies'] = netsuite_prices_df.groupby('Memo')[['Currency']].agg({'Currency':lambda x: ", ".join(list(x))})
    netsuite_prices_df['num_duplicates'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('count')
    all_tables_df['netsuite_prices'] = netsuite_prices_df

    netsuite_agg = netsuite_prices_df.groupby('Memo').agg({'Rate': ['min', 'max', 'count'],
                                                           'Quantity': ['min', 'max'],
                                                           'Currency': lambda x: ", ".join(list(x))})
    netsuite_agg.columns = [' '.join(col).strip() for col in netsuite_agg.columns.values]
    netsuite_agg = netsuite_agg.reset_index()
    netsuite_agg = netsuite_agg.add_suffix('_netsuite')
    all_tables_df['netsuite_agg'] = netsuite_agg #.merge(all_tables_df['wp_type_part'], how='left', left_on='Memo', right_on='name')


def get_wp_tables_by_post_type(all_tables_df):
    all_post_types = list(all_tables_df['wp_posts']['post_type'].unique())
    wp_posts = all_tables_df['wp_posts']
    wp_postmeta = all_tables_df['wp_postmeta']
    for post_type in all_post_types:
        wp_type_posttype = 'wp_type_' + post_type
        wp_posts_post_type = wp_posts[wp_posts['post_type'] == post_type]
        wp_postmeta_post_type = wp_postmeta[(wp_postmeta['post_id'].isin(list(wp_posts_post_type['ID'])))
                                            & (wp_postmeta['meta_key'].str[0] != '_')]
        all_tables_df[wp_type_posttype] = wp_postmeta_post_type.pivot(index='post_id', columns='meta_key', values='meta_value').reset_index()# .drop(columns=['meta_key'])
        all_tables_df[wp_type_posttype] = all_tables_df[wp_type_posttype].merge(wp_posts_post_type, left_on='post_id', right_on='ID').drop(columns=['ID', 'post_type'])

# Dependencies: wp_type_quote (enriched), wp_projects, wp_manufacturers
# Builds pm_project_manufacturer
def pm_project_manufacturer(all_tables_df):
    # training data + labels are based on wp_type_quote
    # Get all project ids that have quotes
    wp_type_quote = all_tables_df['wp_type_quote'][
        ['post_id', 'bids', 'project', 'competing_manufacturers', 'winning_manufacturers']]
    # Join projects features
    wp_projects = all_tables_df['wp_projects']
    pm_df = wp_type_quote.merge(wp_projects, left_on='project', right_on='post_id', suffixes=('_quote', '_project'))
    # For each manufacturer create a project-manufacturer row with data from wp_manufacturers
    wp_manufacturers = all_tables_df['wp_manufacturers'].rename(columns={'post_id': 'post_id_manuf'})
    pm_df = pm_df.merge(wp_manufacturers, how='cross', suffixes=('_quote', '_manuf'))

    # Clean columns data
    standardize_country_values(pm_df)

    # build Label column
    pm_df['is_manuf_bid'] = pm_df.apply(lambda row: 1 if row['post_id_manuf'] in row['competing_manufacturers'] else 0,
                                        axis='columns')

    # set index by project-manufacturer
    pm_df = pm_df.set_index(['post_id_project', 'post_id_manuf'])
    all_tables_df['pm_project_manufacturer'] = pm_df


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
    # print("Writing " + new_table_name)
    # for efficiency
    if new_table_name not in all_tables_df:
        start_table = all_tables_df[pam_table_name]
        start_table[(start_table['cnc_milling'] < start_table['req_milling'])
                    | (start_table['cnc_turning'] < start_table['req_turning'])]['is_manuf_bid'] = 0
        all_tables_df[new_table_name] = start_table.copy()

    return new_table_name


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


