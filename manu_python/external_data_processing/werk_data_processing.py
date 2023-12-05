import json
import os

import pandas as pd
from werk24 import W24Measure
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
    part_name = os.path.basename(results_dir).split(".pdf-Results")[0]
    page_to_data_dict = extract_page_to_data_dict(results_dir)

    list_of_dict_werk_column_name_to_value = []
    for page_dir, data_dict in page_to_data_dict.items():
        page_title_block_column_name_to_value = build_title_block_column_to_value_dict(data_dict, page_dir, part_name,
                                                                                       results_dir)
        list_of_dict_werk_column_name_to_value.append(page_title_block_column_name_to_value)
        for sheet_dir, data_dict in data_dict.items():
            if sheet_dir == 'title_block':
                continue
            for canvas_dir, data_dict in data_dict.items():
                for sectional_dir, data_dict in data_dict.items():
                    sectional_measures_list = data_dict['sectional_measures']
                    for index, measure in enumerate(sectional_measures_list):
                        dict_werk_column_name_to_value = page_title_block_column_name_to_value.copy()
                        dict_werk_column_name_to_value['Sheet'] = sheet_dir
                        dict_werk_column_name_to_value['Canvas'] = canvas_dir
                        dict_werk_column_name_to_value['Sectional'] = sectional_dir
                        dict_werk_column_name_to_value['Item'] = index
                        dict_werk_column_name_to_value['nominal_size'] = float(measure.label.size.nominal_size)
                        # check for null values before casting to float and assigning
                        if measure.label.size_tolerance is None:
                            dict_werk_column_name_to_value['size_tolerance_deviation_lower'] = None
                            dict_werk_column_name_to_value['size_tolerance_deviation_upper'] = None
                        else:
                            if hasattr(measure.label.size_tolerance, 'deviation_lower') and measure.label.size_tolerance.deviation_lower is not None:
                                dict_werk_column_name_to_value['size_tolerance_deviation_lower'] = float(measure.label.size_tolerance.deviation_lower)
                            else:
                                dict_werk_column_name_to_value['size_tolerance_deviation_lower'] = None
                            if hasattr(measure.label.size_tolerance, 'deviation_upper') and measure.label.size_tolerance.deviation_upper is not None:
                                dict_werk_column_name_to_value['size_tolerance_deviation_upper'] = float(measure.label.size_tolerance.deviation_upper)
                            else:
                                dict_werk_column_name_to_value['size_tolerance_deviation_upper'] = None
                        list_of_dict_werk_column_name_to_value.append(dict_werk_column_name_to_value)


    return list_of_dict_werk_column_name_to_value


def build_title_block_column_to_value_dict(data_dict, page_dir, part_name, results_dir):
    page_title_block_column_name_to_value = {'name': part_name
        , 'result_dir_full_path': results_dir
        , 'Page': page_dir}
    title_block = data_dict['title_block']
    if title_block is None or title_block.material is None or title_block.material.material_category is None:
        page_title_block_column_name_to_value['material_categorization_level_1'] = None
        page_title_block_column_name_to_value['material_categorization_level_2'] = None
        page_title_block_column_name_to_value['material_categorization_level_3'] = None
    else:
        page_title_block_column_name_to_value['material_categorization_level_1'] = \
        title_block.material.material_category[0]
        page_title_block_column_name_to_value['material_categorization_level_2'] = \
        title_block.material.material_category[1]
        page_title_block_column_name_to_value['material_categorization_level_3'] = \
        title_block.material.material_category[2]
    return page_title_block_column_name_to_value


def extract_page_to_data_dict(results_dir) -> dict:
    page_to_data_dict = {}
    for page_dir in get_all_directories_with_prefix(results_dir, "Page"):
        full_path_page_dir = os.path.join(results_dir, page_dir)
        page_to_data_dict[page_dir] = {}
        page_to_data_dict[page_dir]['title_block'] = extract_title_block(full_path_page_dir)
        for sheet_dir in get_all_directories_with_prefix(full_path_page_dir, "Sheet"):
            full_path_sheet_dir = os.path.join(full_path_page_dir, sheet_dir)
            page_to_data_dict[page_dir][sheet_dir] = {}
            for canvas_dir in get_all_directories_with_prefix(full_path_sheet_dir, "Canvas"):
                full_path_canvas_dir = os.path.join(full_path_sheet_dir, canvas_dir)
                page_to_data_dict[page_dir][sheet_dir][canvas_dir] = {}
                for sectional_dir in get_all_directories_with_prefix(full_path_canvas_dir, "Sectional"):
                    full_path_sectional_dir = os.path.join(full_path_canvas_dir, sectional_dir)
                    page_to_data_dict[page_dir][sheet_dir][canvas_dir][sectional_dir] = {}
                    sectional_measures_list = []
                    with open(os.path.join(full_path_sectional_dir, "Measure.json")) as json_data:
                        data = json.load(json_data)
                        for i in range(len(data)):
                            measure = W24Measure.parse_obj(data[i])
                            sectional_measures_list.append(measure)
                    page_to_data_dict[page_dir][sheet_dir][canvas_dir][sectional_dir]['sectional_measures'] = sectional_measures_list
    return page_to_data_dict


def get_all_directories_with_prefix(full_path_dir, prefix):
    return [d for d in os.listdir(full_path_dir) if
            d.startswith(prefix) and os.path.isdir(os.path.join(full_path_dir, d))]


def extract_title_block(full_path_page_dir):
    title_block = W24TitleBlock.parse_file(os.path.join(full_path_page_dir, "TitleBlock.json"))
    return title_block


def build_werk_table():
    dal.drop_table('werk')
    dal.create_table('werk')
    dict_werk_column_name_to_type = {
        'name': 'VARCHAR(255)'
        , 'result_dir_full_path': 'VARCHAR(255)'
        , 'Page': 'VARCHAR(255)'
        , 'material_categorization_level_1': 'VARCHAR(255)'
        , 'material_categorization_level_2': 'VARCHAR(255)'
        , 'material_categorization_level_3': 'VARCHAR(255)'
        , 'Sheet': 'VARCHAR(255)'
        , 'Canvas': 'VARCHAR(255)'
        , 'Sectional': 'VARCHAR(255)'
        , 'Item': 'VARCHAR(255)'
        , 'nominal_size': 'FLOAT'
        , 'size_tolerance_deviation_lower': 'FLOAT'
        , 'size_tolerance_deviation_upper': 'FLOAT'
    }
    dal.add_columns_to_table('werk', dict_werk_column_name_to_type)
