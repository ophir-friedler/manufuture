import pandas as pd


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
def clean_wp_quotes(all_tables_df):
    # Remove Avsha's test-agency (216)
    # Remove Ben's test-agency (439)
    test_agencies = ["216", "439"]
    df = all_tables_df['wp_quotes']
    # Fill Nones in empty chosen_bids
    df['chosen_bids'] = df['chosen_bids'].apply(
        lambda chosen_bids: None if (chosen_bids != None and len(chosen_bids) == 0) else chosen_bids)
    df = df.drop(df[df['agency'].isin(test_agencies)].index)
    all_tables_df['wp_quotes'] = df
    return all_tables_df


def clean_tables(all_tables_df):
    clean_wp_manufacturers(all_tables_df)
    clean_wp_quotes(all_tables_df)
    clean_wp_parts(all_tables_df)

