import logging

import matplotlib.pyplot as plt
import pandas as pd

from manu_python.config.config import MANUFACTURER_BID_LABEL_COLUMN_NAME


def display_bids_per_month(bids_df, ax, title):
    g_df = bids_df.groupby([bids_df['post_date_Ym']]).count()
    ax.set_title(title)
    ax.bar(g_df.index, g_df['manufacturer'])


def display_number_of_monthly_bids(all_tables_df, manufacturer_id, ax):
    df = all_tables_df['wp_type_bid']
    manufacturer_bids_df = df[df['manufacturer'].apply(str) == str(manufacturer_id)]
    display_bids_per_month(manufacturer_bids_df, ax, title="ID: " + str(manufacturer_id) + ", Number of bids")


def display_number_of_monthly_chosen_bids(all_tables_df, manufacturer_id, ax):
    df = all_tables_df['wp_type_bid']
    df = df[(df['manufacturer'].apply(str) == str(manufacturer_id)) & (df['is_chosen'] == True)]
    display_bids_per_month(df, ax, title="ID: " + str(
        manufacturer_id) + ", Manufacturer bids that were chosen (i.e.: quote wins): ")


def display_manufacturer_monthly_success_rate(all_tables_df, manufacturer_id, ax):
    df = all_tables_df['wp_type_bid']
    df = df[df['manufacturer'].apply(str) == str(manufacturer_id)]
    df = df.groupby([df['post_date_Ym'], 'is_chosen'])[['post_id']] \
        .count().pivot_table('post_id', ['post_date_Ym'], 'is_chosen').fillna(0)
    if df.empty:
        print("Manufacturer " + str(manufacturer_id) + " has no prior bids")
        return
    if 1 not in df.columns:
        df['success_rate'] = 0
    else:
        df['success_rate'] = 100 * df[1] / (df[1] + df[0])
    ax.set_title("Monthly success rate (%), Manufacturer ID: " + str(manufacturer_id))
    ax.bar(df.index, df['success_rate'])


def manufacturer_dashboard(all_tables_df, manufacturer_id):
    print("Manufacturer ID: " + str(manufacturer_id))
    df = all_tables_df['wp_manufacturers']
    display(df[df['post_id'].apply(str) == str(manufacturer_id)])
    print("Total number of bids: " + str(df[df['post_id'] == manufacturer_id]['participation_count'].values[0]))
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    display_number_of_monthly_bids(all_tables_df, manufacturer_id, axs[0, 0])
    display_number_of_monthly_chosen_bids(all_tables_df, manufacturer_id, axs[0, 1])
    display_manufacturer_monthly_success_rate(all_tables_df, manufacturer_id, axs[1, 0])
    plt.show()


def display_all_manufacturers_monthly_success_rate(all_tables_df, ax):
    df = all_tables_df['wp_type_bid']
    df = df.groupby([df['post_date_Ym'], 'is_chosen'])[['post_id']] \
        .count().pivot_table('post_id', ['post_date_Ym'], 'is_chosen').fillna(0)
    df['success_rate'] = 100 * df[1] / (df[1] + df[0])
    ax.set_title("Monthly total bid success rate (should be replaced with median success rate)")
    ax.bar(df.index, df['success_rate'])


def calc_monthly_success_rate_quantile_df(df, quant):
    return df.groupby(['post_date_Ym'])[['success_rate']].quantile(quant).rename(
        columns={'success_rate': 'success_rate_p_' + str(quant)})


def display_all_manufacturers_monthly_success_rate_distribution(all_tables_df):
    # Calculate success rate
    df = all_tables_df['monthly_bid_success_rate']
    dist_df = calc_monthly_success_rate_quantile_df(df, 0.05)
    for quant in [0.25, 0.5, 0.75, 0.95, 0.99]:
        dist_df = pd.merge(dist_df, calc_monthly_success_rate_quantile_df(df, quant), left_index=True, right_index=True,
                           how="left")
    dist_df.plot.line(figsize=(15, 7), title="Monthly manufacturers bid success rate percentiles")


def manufacturers_high_level_dashboard(all_tables_df):
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    df = all_tables_df['monthly_manufacturers_stats']
    axs[0, 0].set_title("New manufacturers")
    axs[0, 0].bar(df.index, df['num_manufacturers'])

    df = all_tables_df['wp_type_bid']
    display_bids_per_month(df, axs[0, 1], title="Num of manufacturer bids")
    df = df[df['is_chosen'] == True]
    display_bids_per_month(df, axs[1, 0], title="Num of chosen manufacturer bids")
    display_all_manufacturers_monthly_success_rate(all_tables_df, axs[1, 1])
    display_all_manufacturers_monthly_success_rate_distribution(all_tables_df)



    plt.show()


# Display: all_tables_df['monthly_projects_stats']
# Columns: 'num_projects', 'num_projects_with_quote', 'num_projects_approved',
#          'pct_projects_with_quote', 'pct_approved_out_of_with_quote'
def projects_high_level_dashboard(all_tables_df):
    df = all_tables_df['monthly_projects_stats']
    fig, axs = plt.subplots(3, 2, figsize=(16, 15))
    axs[0, 0].set_title("Created projects")
    axs[0, 0].bar(df.index, df['num_projects'])

    axs[0, 1].set_title("Num projects with a quote (by creation date)")
    axs[0, 1].bar(df.index, df['num_projects_with_quote'])

    axs[1, 0].set_title("Approved projects")
    axs[1, 0].bar(df.index, df['num_projects_approved'])

    axs[1, 1].set_title("Percent of projects with quote")
    axs[1, 1].bar(df.index, df['pct_projects_with_quote'])

    axs[2, 0].set_title("Percent approved projects out of projects with quote")
    axs[2, 0].bar(df.index, df['pct_approved_out_of_with_quote'])
    plt.show()


def agency_dashboard(all_tables_df, agency_id):
    agency_id_str = str(agency_id)
    print("Agency ID: " + agency_id_str)
    wp_projects_df = all_tables_df['wp_projects']
    wp_projects_agency_df = wp_projects_df[wp_projects_df['agency'].apply(str) == str(agency_id)]
    print("Number of projects: " + str(len(wp_projects_agency_df)))
    pm = all_tables_df['pm_project_manufacturer']
    pm_agency_bid_manufs = pm[(pm['agency'] == agency_id_str) & (pm[MANUFACTURER_BID_LABEL_COLUMN_NAME] > 0)]
    agency_manufs_bid = pm_agency_bid_manufs.reset_index().groupby('agency')[['post_id_manuf', MANUFACTURER_BID_LABEL_COLUMN_NAME]]
    uniques_df = agency_manufs_bid.nunique()
    # check that uniques_df is not empty (i.e.: agency has bids)
    if uniques_df.empty:
        logging.error("Agency " + agency_id_str + " has no bids")
        return
    num_manufs_bid_on_agency = uniques_df.at[agency_id_str, 'post_id_manuf']
    print("Number of manufacturers that bid for agency's projects: " + str(num_manufs_bid_on_agency))
    print("from wp_projects: ")
    display(wp_projects_df[wp_projects_df['agency'].apply(str) == agency_id_str])

    # fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    # display_number_of_monthly_bids(all_tables_df, manufacturer_id, axs[0, 0])
    # display_number_of_monthly_chosen_bids(all_tables_df, manufacturer_id, axs[0, 1])
    # display_manufacturer_monthly_success_rate(all_tables_df, manufacturer_id, axs[1, 0])
    # plt.show()


# Calculate # of quotes with/without a chosen bid per number of candidates in the quote
def display_auction_success_rate_by_num_candidates(all_tables_df):
    df = all_tables_df['stats_by_num_candidates']
    print("Auction success rate by num candidates:")
    print("X axis: num candidates per quote")
    print("Y axis: % of quotes with chosen bid")

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(df.index,df['success_rate'])
    plt.show()
    display(df)


# Participation count per manufacturer
def display_manufacturers_participation_distribution(all_tables_df):
    n, bins, patches = plt.hist(all_tables_df['wp_manufacturers']['participation_count'], 60, facecolor='blue', alpha=0.5)
    plt.title("Histogram of participation count per manufacturer")
    plt.xlabel('participation_count')
    plt.ylabel('number of such manufacturers')
    plt.show()
