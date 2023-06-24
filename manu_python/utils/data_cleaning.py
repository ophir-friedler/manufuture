import logging

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


# wp_quotes cleaning:
def clean_quotes_tables(all_tables_df):
    # Remove Avsha's test-agency (216)
    # Remove Ben's test-agency (439)
    test_agencies = ["216", "439"]
    for table_name in ['wp_quotes', 'wp_type_quote']:
        logging.warning('table_name:' + table_name)
        df = all_tables_df[table_name]
        # Fill Nones in empty chosen_bids
        df['chosen_bids'] = df['chosen_bids'].apply(
            lambda chosen_bids: None if (chosen_bids is None or chosen_bids != chosen_bids or str(chosen_bids) == "") else chosen_bids)
        df = df.drop(df[df['agency'].isin(test_agencies)].index)
        all_tables_df[table_name] = df
    return all_tables_df


table_column_pairs_containing_arrays = [('wp_type_quote', 'bids'), ('wp_type_quote', 'chosen_bids')]


def digit_array_of_digits_transform(digit_or_string):
    if digit_or_string is None or (isinstance(digit_or_string, str) and len(digit_or_string) == 0):
        return []
    if digit_or_string.isdigit():
        return digit_or_string
    if isinstance(digit_or_string, str):
        return [int(x) for x in dict_to_list(loads(str.encode(digit_or_string)))]
    logging.error("Should not ever reach nere, parsing error in some table, column")


def clean_wp_type_tables(all_tables_df):
    for table, column in table_column_pairs_containing_arrays:
        all_tables_df[table][column] = all_tables_df[table][column].apply(digit_array_of_digits_transform)

                # [int(x) for x in dict_to_list(loads(b'a:2:{i:0;s:5:"14874";i:1;s:5:"15001";}'))]


def clean_tables(all_tables_df):
    clean_wp_manufacturers(all_tables_df)
    clean_quotes_tables(all_tables_df)
    clean_wp_parts(all_tables_df)
    clean_wp_type_tables(all_tables_df)

