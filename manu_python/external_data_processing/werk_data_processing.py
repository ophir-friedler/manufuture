import os

import pandas as pd
from werk24.models.title_block import W24TitleBlock

# Print all names of pdf files in a werk directory
from manu_python.db import dal


# process all werk results directories and write them to manufuture database werk table
def process_all_werk_results_dirs_to_df(starting_dir) -> list:
    results_dirs = get_all_werk_results_dirs(starting_dir)
    all_results_list = []
    for results_dir in results_dirs:
        list_of_dict_werk_column_name_to_value = extract_features_form_werk_results_dir(results_dir)
        for dict_werk_column_name_to_value in list_of_dict_werk_column_name_to_value:
            all_results_list = all_results_list + [dict_werk_column_name_to_value]
            # dal.insert_row_to_table('werk', dict_werk_column_name_to_value)
    return all_results_list


# get all result directories from a starting directory
def get_all_werk_results_dirs(starting_dir):
    # get list of all directories with suffix "Results" in starting_dir recursively
    results_dirs = []
    for root, dirs, files in os.walk(starting_dir):
        for dir in dirs:
            if dir.endswith("Results"):
                results_dirs.append(os.path.join(root, dir))

    return results_dirs


def extract_features_form_werk_results_dir(results_dir):
    name = os.path.basename(results_dir).split(".pdf-Results")[0]
    page_dirs = [page_dir for page_dir in os.listdir(results_dir) if page_dir.startswith("Page")]
    page_to_title_block = {}
    for page_dir in page_dirs:
        full_path_dir = os.path.join(results_dir, page_dir)
        title_block = W24TitleBlock.parse_file(os.path.join(full_path_dir, "TitleBlock.json"))
        page_to_title_block[page_dir] = title_block

    list_of_dict_werk_column_name_to_value = []
    for page_dir, title_block in page_to_title_block.items():
        if title_block is None or title_block.material is None or title_block.material.material_category is None:
            dict_werk_column_name_to_value = {'name': name
                , 'dir_full_path': results_dir
                , 'page_number': page_dir
                , 'material_categorization_level_1': None
                , 'material_categorization_level_2': None
                , 'material_categorization_level_3': None
                                              }
        else:
            dict_werk_column_name_to_value = {'name': name
                , 'dir_full_path': results_dir
                , 'page_number': page_dir
                , 'material_categorization_level_1': title_block.material.material_category[0]
                , 'material_categorization_level_2': title_block.material.material_category[1]
                , 'material_categorization_level_3': title_block.material.material_category[2]
                                          }
        list_of_dict_werk_column_name_to_value.append(dict_werk_column_name_to_value)
    return list_of_dict_werk_column_name_to_value


def build_werk_table():
    dal.drop_table('werk')
    dal.create_table('werk')
    dict_werk_column_name_to_type = {
        'name': 'VARCHAR(255)'
        , 'dir_full_path': 'VARCHAR(255)'
        , 'page_number': 'VARCHAR(255)'
        , 'material_categorization_level_1': 'VARCHAR(255)'
        , 'material_categorization_level_2': 'VARCHAR(255)'
        , 'material_categorization_level_3': 'VARCHAR(255)'
    }
    dal.add_columns_to_table('werk', dict_werk_column_name_to_type)
