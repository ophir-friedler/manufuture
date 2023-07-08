import logging
import math

import pandas as pd
from phpserialize import dict_to_list, loads


# Clean wp_manufacturers
# Remove test manufacturers
# Replace nans with 0
def clean_wp_manufacturers(all_tables_df):
    # remove test manufacturers
    test_manufacturers = [437, 590, 708, 1268, 24840]
    df = all_tables_df['wp_manufacturers']
    df = df.drop(df[df['post_id'].isin(test_manufacturers)].index)
    all_tables_df['wp_manufacturers'] = df
    # replace Empty strings in 'vendors' with 0
    all_tables_df['wp_manufacturers']['vendors'].replace('', 0, inplace=True)
    # replace Nons with 0:
    for nan_col in ['conventional_milling', 'conventional_turning', 'sheet_metal_press_break', 'sheet_metal_punching',
                    'sheet_metal_weldings', 'preffered_type_full_turnkey', 'preffered_type_assemblies']:
        all_tables_df['wp_manufacturers'][nan_col].fillna('0', inplace=True)
    # translate vendors type to int
    all_tables_df['wp_manufacturers']['vendors'] = all_tables_df['wp_manufacturers']['vendors'].astype('int64')
    all_tables_df['wp_manufacturers']['cnc_milling'] = all_tables_df['wp_manufacturers']['cnc_milling'].fillna(0).astype('int64')
    all_tables_df['wp_manufacturers']['cnc_turning'] = all_tables_df['wp_manufacturers']['cnc_turning'].fillna(0).astype('int64')


def clean_wp_parts(all_tables_df):
    df = all_tables_df['wp_parts']

    # Replace empty strings with NaN values in the 'quantity' column and then fill with zeros
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

    # Expand machining process
    df = pd.concat([df, pd.get_dummies(df['machining_process'])], axis=1)

    all_tables_df['wp_parts'] = df
    return all_tables_df


def get_bids_from_row(bids_from_row) -> list:
    if bids_from_row is None:
        return []
    if isinstance(bids_from_row, list):
        return bids_from_row
    if bids_from_row.isdigit():
        return [int(bids_from_row)]
    if isinstance(bids_from_row, str):
        if len(bids_from_row.strip()) == 0:
            return []
        bids_split = bids_from_row[1:-1].split(",")
        return [int(bid) for bid in bids_split]


def clean_wp_type_quote(all_tables_df):
    df = all_tables_df['wp_type_quote']
    # Remove Avsha's test-agency (216)
    # Remove Ben's test-agency (439)
    df = df.drop(df[df['agency'].isin(["216", "439"])].index)
    df['bids'] = df['bids'].apply(get_bids_from_row)
    df['chosen_bids'] = df['chosen_bids'].apply(get_bids_from_row)
    all_tables_df['wp_type_quote'] = df
    return all_tables_df


def clean_wp_type_bid(all_tables_df):
    all_tables_df['wp_type_bid']['manufacturer'] = all_tables_df['wp_type_bid']['manufacturer'].astype('int64')


def digit_array_of_digits_transform(digit_or_string):
    if (digit_or_string is None) or \
            (isinstance(digit_or_string, float) and math.isnan(digit_or_string)) or \
            (isinstance(digit_or_string, str) and len(digit_or_string) == 0):
        return []
    if digit_or_string.isdigit():
        return digit_or_string
    if isinstance(digit_or_string, str):
        return [int(x) for x in dict_to_list(loads(str.encode(digit_or_string)))]
    logging.error("Should not ever reach nere, parsing error in some table, column")


def clean_wp_type_tables(all_tables_df):
    for table, column in [('wp_type_quote', 'bids'), ('wp_type_quote', 'chosen_bids')]:
        all_tables_df[table][column] = all_tables_df[table][column].apply(digit_array_of_digits_transform)


def clean_tables(all_tables_df):
    clean_wp_type_tables(all_tables_df)
    clean_wp_manufacturers(all_tables_df)
    clean_wp_type_quote(all_tables_df)
    clean_wp_type_bid(all_tables_df)
    clean_wp_parts(all_tables_df)


