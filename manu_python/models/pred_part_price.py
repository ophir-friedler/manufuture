import itertools
import logging

import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

from manu_python.config.config import STATIC_DATA_DIR_PATH, PART_NETSUITE_MEAN_PRICE_COLUMN_NAME


class PartPricePredictor:
    _label_column = PART_NETSUITE_MEAN_PRICE_COLUMN_NAME
    _categorical_features = None
    _input_table_name = 'part_price_training_table'
    _training_table_name = _input_table_name + '_training'
    # _manufacturers_data_df = None
    _is_model_trained = False
    _model_types = ['deep_regress_v0']
    _model_type = None
    _model = None

    _all_part_features = [
        # ## Part features
        'coc'
        , 'first_material_categorization_level_1_list'
        , 'first_material_categorization_level_2_list'
        , 'first_material_categorization_level_3_list'

    ]

    _selected_singles = [
        # ## Part features
        'coc'
        , 'first_material_categorization_level_1_list'
        , 'first_material_categorization_level_2_list'
        , 'first_material_categorization_level_3_list'

    ]
    _selected_doubles = [
        # ['post_id_manuf', 'plan']
        # , ['post_id_manuf', 'req_sheet_metal_inserts']
        # , ['post_id_manuf', 'req_sheet_metal']
        # , ['post_id_manuf', 'one_manufacturer']
        # , ['post_id_manuf', 'num_distinct_parts_binned']
        # , ['post_id_manuf', 'total_quantity_of_parts_binned']
        # , ['sheet_metal_weldings', 'sheet_metal_punching']
    ]

    def __init__(self):
        # Singles
        self._x_train_two_rows = None
        self._all_used_features =  self._all_part_features
        # Feature selection
        self._all_training_features = self._selected_singles + self.get_selected_double_feature_names()
        # Categorical
        self._categorical_features = self._all_training_features

    def __str__(self):
        ret_str = "_input_table_name: " + self._input_table_name + '\n' + " _label_column: " + self._label_column \
                  + " _training_table_name: " + self._training_table_name \
                  + " _model: " + str(self._model)
        return ret_str

    def save_model(self):
        model_name = 'model__' + self._model_type + '__' + 'T__' + self._training_table_name
        model_save_path = STATIC_DATA_DIR_PATH + model_name + '.h5'
        self._model.save(model_save_path)
        # save the fit_predict_columns to a file and maintain the order of the columns
        self._x_train_two_rows.to_parquet(STATIC_DATA_DIR_PATH + 'x_train_two_rows.parquet')

        logging.warning("Saved model to: " + model_save_path)

    def load_model(self, model_path):
        self._model = tf.keras.models.load_model(model_path)
        # extract model type from model name, the name has the format model__<model_type>__T__<training_table_name>
        self._model_type = model_path.split('__')[1]
        self._is_model_trained = True
        self._x_train_two_rows = pd.read_parquet(STATIC_DATA_DIR_PATH + 'x_train_two_rows.parquet')

    def _validate_configuration(self):
        all_features_set = set(self._selected_singles).union(set([single for double in self._selected_doubles for single in double]))
        sym_dif = set(self._all_used_features).symmetric_difference(all_features_set)
        if len(sym_dif) > 0:
            logging.error("inconsistency in the following features : " + str(sym_dif))
            return False
        return True

    # model_types: in self._model_types
    def build_model(self, all_tables_df, model_type, verbose=False):
        self._model_type = model_type
        if self._validate_configuration():
            # self._manufacturers_data_df = all_tables_df[self._input_table_name][self._all_manufacturer_features].drop_duplicates()
            all_tables_df[self._training_table_name] = self.prepare_for_fit_predict(
                all_tables_df[self._input_table_name], verbose=verbose)
            if verbose:
                print("Training data table name: " + self._training_table_name)
            self.train_part_price_predictor(training_data=all_tables_df[self._training_table_name],
                                            model_type=model_type,
                                            verbose=verbose)
            return self._model

    def get_selected_double_feature_names(self):
        return [get_double_feature_name(column_a, column_b) for [column_a, column_b] in self._selected_doubles]

    def prepare_doubles(self, raw_data):
        for [column_a, column_b] in self._selected_doubles:
            raw_data[get_double_feature_name(column_a, column_b)] = raw_data.apply(
                lambda row: get_double_feature_value_new(row, column_a, column_b), axis='columns')
        return raw_data

    def do_all_features_exist(self, columns):
        cols_difference_set = set(self._all_used_features).difference(set(columns))
        if len(cols_difference_set) > 0:
            logging.error("missing columns: " + str(cols_difference_set))
            return False
        return True

    def model_predict(self, predict_input):
        if self._model_type == 'deep_regress_v0':
            logging.info("Predicting with model: " + str(self._model))
            return self._model.predict(predict_input, verbose=0)
        logging.error("Shouldn't reach here, model type unkown: " + self._model_type)
        return None

    # Training data needs to contain all features, as well as target feature
    def train_part_price_predictor(self, training_data, model_type, verbose=False):
        X_train = training_data.drop(columns=[self._label_column])
        y_train = training_data[self._label_column]

        # shuffling
        # train_test_split(training_data, y_train, test_size=1, random_state=1)

        # train
        if model_type is None:
            logging.error("Model type (" + str(self._model_types) + "), not selected")
        elif model_type not in self._model_types:
            logging.error("Unknown model type: " + model_type + ", please select from: " + str(self._model_types))
        elif model_type == 'deep_regress_v0':  # Create deep regression model using Keras
            model = Sequential()
            model.add(Dense(64, activation='relu', input_dim=len(X_train.columns)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='linear'))

            # Compile the model
            model.compile(optimizer='adam',
                          loss='mean_squared_error')
            if verbose:
                print("Model: " + str(model))
            model.fit(X_train, y_train, epochs=10, batch_size=32)
            self._model = model
            self._is_model_trained = True
            self._x_train_two_rows = X_train.head(2)

    # Predict price for a single part based on its features
    def predict_part_price(self, part_features_map):
        row = pd.DataFrame.from_dict(part_features_map)
        prepared_row = self.prepare_for_fit_predict(row)
        return self.model_predict(prepared_row)

    def price_predictions_for_all_feature_combinations(self, all_tables_df, csv_filename):
        # collect all feature values for all features, and then create a cartesian product of all feature values
        all_values = {}
        for feature in self._all_part_features:
            all_values[feature] = all_tables_df[self._input_table_name][feature].unique()

        ret_df = pd.DataFrame()
        for element in itertools.product(*[all_values[feature] for feature in self._all_part_features]):
            part_features_map = {k: [v] for k, v in zip(self._all_part_features, element)}
            predict_rows = self.predict_part_price(part_features_map)
            predict_rows = pd.DataFrame(predict_rows, columns=['pred_price'])
            predict_rows = pd.concat([pd.DataFrame(part_features_map), predict_rows], axis=1)
            ret_df = pd.concat([ret_df, predict_rows])
        ret_df.to_csv(csv_filename)


    def add_manufacturers_columns_to_predict_rows(self, all_tables_df, predict_rows, manufacturers_extra_columns):
        manuf_name_df = all_tables_df['wp_manufacturers'][['post_id'] + manufacturers_extra_columns]
        predict_rows = predict_rows.merge(manuf_name_df, left_on='post_id_manuf', right_on='post_id').drop(
            columns=['post_id'])
        predict_rows['separator'] = ''
        predict_rows = predict_rows[
            self._all_project_features + ['separator', 'predBidProb', 'post_id_manuf'] + manufacturers_extra_columns]
        return predict_rows

    # Complete columns required for predicting from model (if missing then should be 0 in a 1-hot encoding)
    def complete_columns_with_negatives(self, prepared_data):
        if self._is_model_trained is True and self._x_train_two_rows is not None:
            missing_columns = list(set(self._x_train_two_rows.columns).difference(set(prepared_data.columns)))
            prepared_data = pd.concat([prepared_data, pd.DataFrame(index=prepared_data.index, columns=missing_columns)],
                                      axis=1)
            prepared_data = prepared_data.fillna(0)

            # Reorder (and filter) columns to the order at train time
            prepared_data = prepared_data[self._x_train_two_rows.columns]
        else:
            logging.error("Trying to complete columns before model is trained")
        return prepared_data

    # Feature validation
    # Only necessary features
    # One hot encoding
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
                print("Label column")
                print(self._label_column)
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
        logging.error("Features missing in data passed to prepare_for_fit_predict")
        return pd.DataFrame()


def get_double_feature_name(column_a, column_b):
    return column_a + "__" + column_b


def get_double_feature_value(df, column_a, column_b):
    return df[column_a].astype("string") + "_" + \
           df[column_b].astype("string")


def get_double_feature_value_new(ser, column_a, column_b):
    return str(ser[column_a]) + "_" + str(ser[column_b])
