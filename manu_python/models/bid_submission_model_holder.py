import json

import pandas as pd
import tensorflow as tf


class BidSubmissionPredictorHolder:
    _config = None
    _model = None
    _manufacturers_data_df = None
    _model_type = None
    _x_train_two_rows = None

    def __init__(self, config_file_path=None):
        with open(config_file_path, 'r') as file:
            print("Loading config from file: " + config_file_path)
            self._config = json.load(file)
        self._all_manufacturer_features = self._config['ALL_MANUFACTURER_FEATURES']
        self._all_project_features = self._config['ALL_PROJECT_FEATURES']
        self._selected_singles = self._config['SELECTED_SINGLES']
        self._selected_doubles = self._config['SELECTED_DOUBLES']
        self._label_column = self._config['LABEL_COLUMN']
        self._all_used_features = self._all_manufacturer_features + self._all_project_features
        self._all_training_features = self._selected_singles + self.get_selected_double_feature_names()
        self._categorical_features = self._all_training_features

    def get_selected_double_feature_names(self):
        return [get_double_feature_name(column_a, column_b) for [column_a, column_b] in self._selected_doubles]

    def load_model(self):
        self._model = tf.keras.models.load_model(self._config['STATIC_DATA_DIR_PATH'] + self._config['MODEL_FILE_NAME'])
        self._manufacturers_data_df = pd.read_parquet(self._config['STATIC_DATA_DIR_PATH']
                                                      + self._config['MANUFACTURERS_DATA_DF_FILE_NAME'])
        # extract model type from model name, the name has the format model__<model_type>__T__<training_table_name>
        self._model_type = self._config['MODEL_FILE_NAME'].split('__')[1]
        self._x_train_two_rows = pd.read_parquet(self._config['STATIC_DATA_DIR_PATH'] + self._config['X_TRAIN_TWO_ROWS_FILE_NAME'])

    def enrich_with_manufacturers_features(self, project_features_map):
        row = pd.DataFrame.from_dict(project_features_map)
        manuf_features_df = self._manufacturers_data_df
        ret_df = row.merge(manuf_features_df, how='cross')
        return ret_df

    def do_all_features_exist(self, columns):
        cols_difference_set = set(self._all_used_features).difference(set(columns))
        if len(cols_difference_set) > 0:
            print("missing columns: " + str(cols_difference_set))
            return False
        return True

    def prepare_for_fit_predict(self, input_table_rows_df, verbose=False):
        # Feature validation
        if self.do_all_features_exist(input_table_rows_df.columns):
            ret_df = input_table_rows_df.copy()
            ret_df = self.prepare_doubles(ret_df)
            if verbose:
                print("Before one hot encoding: ")
                print(len(list(ret_df.columns)))
                print(list(ret_df.columns))
                print("All training features: ")
                print(self._all_training_features)
                print("Categorical: ")
                print(self._categorical_features)

            # Only necessary features
            ret_df = ret_df[ret_df.columns.intersection(self._all_training_features + [self._label_column])]
            if verbose:
                print("ret_df after intersection")
                print(len(list(ret_df.columns)))
                print(list(ret_df.columns))

            # One hot encoding
            for categorical_feature in self._categorical_features:
                if categorical_feature in ret_df.columns:
                    ret_df = pd.concat([ret_df, pd.get_dummies(ret_df[categorical_feature],
                                                               prefix=categorical_feature)],
                                       axis=1).drop(columns=[categorical_feature])
            ret_df = self.complete_columns_with_negatives(prepared_data=ret_df)
            return ret_df
        print("Features missing in data passed to prepare_for_fit_predict")
        return pd.DataFrame()

    def rank_manufacturers_for_project_features(self, project_features_map, verbose=False):
        predict_rows = self.enrich_with_manufacturers_features(project_features_map)
        if verbose:
            print("predict_rows columns: \n" + str(list(predict_rows.columns)))
            print("predict_rows: \n" + str(predict_rows))
        prepared_rows = self.prepare_for_fit_predict(predict_rows)
        if verbose:
            print("prepared_rows columns: \n" + str(list(prepared_rows.columns)))
            print("prepared_rows: \n" + str(prepared_rows))
        predictions_arr = self._model.predict(prepared_rows, verbose=0)

        # Add predictions column from model
        prepared_rows['predBidProb'] = predictions_arr
        predict_rows['predBidProb'] = predictions_arr

        return prepared_rows, predict_rows.sort_values(by=['predBidProb'], ascending=False)

    def prepare_doubles(self, raw_data):
        for [column_a, column_b] in self._selected_doubles:
            raw_data[get_double_feature_name(column_a, column_b)] = raw_data.apply(
                lambda row: get_double_feature_value_new(row, column_a, column_b), axis='columns')
        return raw_data

    # Complete columns required for predicting from model (if missing then should be 0 in a 1-hot encoding)
    def complete_columns_with_negatives(self, prepared_data):
        missing_columns = list(set(self._x_train_two_rows.columns).difference(set(prepared_data.columns)))
        prepared_data = pd.concat([prepared_data, pd.DataFrame(index=prepared_data.index, columns=missing_columns)],
                                  axis=1)
        prepared_data = prepared_data.fillna(0)

        # Reorder (and filter) columns to the order at train time
        prepared_data = prepared_data[self._x_train_two_rows.columns]
        return prepared_data


def get_double_feature_name(column_a, column_b):
    return column_a + "__" + column_b


def get_double_feature_value_new(ser, column_a, column_b):
    return str(ser[column_a]) + "_" + str(ser[column_b])
