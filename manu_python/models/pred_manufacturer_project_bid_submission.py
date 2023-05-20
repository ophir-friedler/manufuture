import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
# import tensorflow as tf
import itertools
from keras.models import Sequential
from keras.layers import Dense


from data_pipelines import table_builder

MIN_NUM_MANUFACTURER_BIDS = 4


class BidSubmissionPredictor:
    # The actual table is all_tables_df[_input_table_name]
    _input_table_name = None
    _label_column = None
    _categorical_features = None
    _training_table_name = 'experiment_1_training_data'
    _is_model_trained = False
    _fit_predict_columns = None

    _model_types = ['lr', 'deep_v0']
    _model_type = None

    _model = None
    _all_manufacturer_features = [
        # ## Manufacturer features
        'post_id_manuf',
        # 'manufacturer_name',
        'sheet_metal_inserts',
        'sheet_metal_weldings',
        'sheet_metal_punching',
        # 'agency',
        'employees_num',
        # 'international',
        'country',
        # 'cnc_turning',
    ]
    _all_project_features = [
        # ## Project features
        'req_turning'
        , 'req_milling'
        , 'plan'
        , 'req_sheet_metal'
        , 'req_sheet_metal_inserts'
        , 'one_manufacturer'
        # , 'num_distinct_parts'
        , 'num_distinct_parts_binned'
        # 'req_batches'
        , 'total_quantity_of_parts_binned'
    ]
    _all_single_features = _all_manufacturer_features + _all_project_features

    _singles = [
        # ## Manufacturer features
        # 'post_id_manuf',
        # 'agency',
        'employees_num'
        , 'sheet_metal_inserts'
        # , 'international'
        , 'country'
        # , 'cnc_turning',

        # ## Project features
        , 'req_turning'
        , 'req_milling'
        , 'plan'
        , 'req_sheet_metal'
        , 'req_sheet_metal_inserts'
        # , 'num_distinct_parts'
        # , 'one_manufacturer'
        # , 'req_batches'
    ]
    _doubles = [
        ['post_id_manuf', 'plan']
       , ['post_id_manuf', 'req_sheet_metal_inserts']
       , ['post_id_manuf', 'req_sheet_metal']
       , ['post_id_manuf', 'one_manufacturer']
       , ['post_id_manuf', 'num_distinct_parts_binned']
       , ['post_id_manuf', 'total_quantity_of_parts_binned']
       , ['sheet_metal_weldings', 'sheet_metal_punching']
    ]

    _all_training_features = []

    def __str__(self):
        ret_str = "_input_table_name: " + self._input_table_name + '\n' + " _label_column: " + self._label_column \
            + " _training_table_name: " + self._training_table_name \
            + " _model: " + str(self._model)
        return ret_str

    def _validate_configuration(self):
        all_features_set = set(self._singles).union(set([single for double in self._doubles for single in double]))
        sym_dif = set(self._all_single_features).symmetric_difference(all_features_set)
        if len(sym_dif) > 0:
            logging.error("inconsistency in the following features : " + str(sym_dif))
            return False
        return True

    # model_type: 'deep_v0' or 'lr'
    def build_model(self, all_tables_df, label_column, model_type, verbose=False):
        self._model_type = model_type
        if self._validate_configuration():
            self._input_table_name = table_builder.build_proj_manu_training_table(all_tables_df,
                                                                                  MIN_NUM_MANUFACTURER_BIDS)
            self._label_column = label_column

            # Feature selection
            self._all_training_features = self._singles + self.get_all_double_feature_names()

            # Categorical
            self._categorical_features = self._all_training_features

            all_tables_df[self._training_table_name] = self.prepare_for_fit_predict(
                all_tables_df[self._input_table_name])
            if verbose:
                print("Training data table name: " + self._training_table_name)
            self.train_bid_submission_predictor(training_data=all_tables_df[self._training_table_name],
                                                              label_column=self._label_column,
                                                              model_type=model_type,
                                                              verbose=verbose)
            return self._model

    def get_all_double_feature_names(self):
        return [get_double_feature_name(column_a, column_b) for [column_a, column_b] in self._doubles]

    def prepare_doubles(self, raw_data):
        for [column_a, column_b] in self._doubles:
            raw_data[get_double_feature_name(column_a, column_b)] = raw_data.apply(
                lambda row: get_double_feature_value_new(row, column_a, column_b), axis='columns')
        return raw_data

    def do_all_features_exist(self, columns):
        cols_difference_set = set(self._all_single_features).difference(set(columns))
        if len(cols_difference_set) > 0:
            logging.error("missing columns: " + str(cols_difference_set))
            return False
        return True

    def model_predict(self, predict_input):
        if self._model_type == 'lr':
            return self._model.predict_proba(predict_input)[:, 1]
        if self._model_type == 'deep_v0':
            return self._model.predict(predict_input, verbose=0)
        logging.error("Shouldn't reach here, model type unkown: " + self._model_type)
        return None

    # Returns lr_model trained on training_data
    # Training data needs to contain all features, as well as target feature
    def train_bid_submission_predictor(self, training_data, label_column, model_type, verbose=False):
        X_train = training_data.drop(columns=[label_column])
        y_train = training_data[label_column]

        # shuffling
        # train_test_split(training_data, y_train, test_size=1, random_state=1)

        # train
        if model_type is None:
            logging.error("Model type (" + str(self._model_types) + "), not selected")
        elif model_type not in self._model_types:
            logging.error("Unknown model type: " + model_type + ", please select from: " + str(self._model_types))
        else:
            if model_type == 'lr':
                if verbose:
                    print("Model: LogisticRegression(max_iter=300)")
                lr_model = LogisticRegression(max_iter=300)
                lr_model.fit(X_train, y_train)
                self._model = lr_model
                self._is_model_trained = True
                self._fit_predict_columns = X_train.columns
            elif model_type == 'deep_v0': # Create deep model using Keras
                model = Sequential()
                model.add(Dense(64, activation='relu', input_dim=len(X_train.columns)))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))

                # Compile the model
                model.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
                if verbose:
                    print("Model: " + str(model))
                model.fit(X_train, y_train, epochs=10, batch_size=32)
                self._model = model
                self._is_model_trained = True
                self._fit_predict_columns = X_train.columns


    def predict_on_proj_manuf(self, all_tables_df, project_id, manufacturer_id, verbose=False):
        if self._is_model_trained is True:
            df = all_tables_df[self._input_table_name]
            row = df[(df['post_id_project'] == project_id) & (df['post_id_manuf'] == manufacturer_id)]
            if len(row) > 1:
                logging.error(
                    "project id: " + str(project_id) + ", manufacturer id:" + str(
                        manufacturer_id) + " too many rows" + str(
                        row))
                return
            if len(row) == 0:
                logging.error(
                    "project id: " + str(project_id) + ", manufacturer id:" + str(manufacturer_id) + " no row found")
                return

            prepared_row = self.prepare_for_fit_predict(row)
            # prepared_row = self.complete_columns_with_negatives(prepared_row)
            if verbose:
                print(row)
            # predictions_arr = self._model.predict_proba(prepared_row)
            predictions_arr = self.model_predict(prepared_row)
            return prepared_row, predictions_arr
        logging.error("Trying to predict before model is trained")

    # Returns a dataframe with _input_table_name rows with:
    # # 'post_id_project', 'post_id_manuf',
    # # single features,
    # # predBidProb - prediction that manufacturer will bid on project
    def rank_manufacturers_for_project(self, all_tables_df, project_id, verbose=False):
        if verbose:
            print("input table name: " + self._input_table_name)
        df = all_tables_df[self._input_table_name]
        rows = df[(df['post_id_project'] == project_id)]
        prepared_rows = self.prepare_for_fit_predict(rows, verbose)
        # prepared_rows = self.complete_columns_with_negatives(prepared_rows)

        # predictions_arr = self._model.predict_proba(prepared_rows)
        if verbose:
            print("After prepare for fit predict")
            print(len(list(prepared_rows.columns)))
            print(list(prepared_rows.columns))

        predictions_arr = self.model_predict(prepared_rows)

        if verbose:
            print("Predictions arr")
            print(predictions_arr)

        ret_df = rows[
            self._singles + ['post_id_project', 'post_id_manuf', 'competing_manufacturers', 'is_manuf_bid']].copy()
        ret_df['predBidProb'] = predictions_arr
        # return rows[['post_id_project', 'post_id_manuf']], self._model.predict_proba(prepared_rows)
        return ret_df.sort_values(by=['predBidProb'], ascending=False)

    # Returns two dataframes with equal number of rows:
    # 1. prepared rows (rows that the model knows how to predict on)
    # 2. predicted rows (human readable rows with the predictions on them
    # Columns of predicted rows: project features (from map), manufacturer features
    def rank_manufacturers_for_project_features(self, all_tables_df, project_features_map):
        predict_rows = self.enrich_with_manufacturers_features(all_tables_df, project_features_map)
        prepared_rows = self.prepare_for_fit_predict(predict_rows)
        # predictions_arr = self._model.predict_proba(prepared_rows)
        predictions_arr = self.model_predict(prepared_rows)

        # Add predictions column from model
        prepared_rows['predBidProb'] = predictions_arr
        predict_rows['predBidProb'] = predictions_arr

        # # augment with manufacturer data
        # manuf_name_df = all_tables_df['wp_manufacturers'][['post_id', 'manufacturer_name']]
        # predict_rows = predict_rows.merge(manuf_name_df, left_on='post_id_manuf', right_on='post_id').drop(columns=['post_id'])
        return prepared_rows, predict_rows.sort_values(by=['predBidProb'], ascending=False)

    def enrich_with_manufacturers_features(self, all_tables_df, project_features_map):
        row = pd.DataFrame.from_dict(project_features_map)
        manuf_features_df = all_tables_df[self._input_table_name][self._all_manufacturer_features].drop_duplicates()
        ret_df = row.merge(manuf_features_df, how='cross')
        return ret_df

    def rank_for_all_projects_to_csv(self, all_tables_df, max_num_recommendations, csv_filename):
        project_ids = set(all_tables_df[self._input_table_name]['post_id_project'])
        ret_df = pd.DataFrame()
        for project_id in project_ids:
            ret_df = pd.concat([ret_df, self.rank_manufacturers_for_project(all_tables_df, project_id).head(max_num_recommendations)])
        ret_df.to_csv(csv_filename)

    def rank_for_all_project_features_to_csv(self, all_tables_df, max_recommendations,
                                             should_filter_out_pending_vendors,
                                             csv_filename, verbose=False):
        all_values = {}
        for feature in self._all_project_features:
            all_values[feature] = all_tables_df[self._input_table_name][feature].unique()
        ret_df = pd.DataFrame()
        for element in itertools.product(*[all_values[feature] for feature in self._all_project_features]):
            project_features_map = {k: [v] for k, v in zip(self._all_project_features, element)}
            if verbose:
                print(project_features_map)
            _, predict_rows = self.rank_manufacturers_for_project_features(all_tables_df, project_features_map)

            predict_rows = self.add_manufacturers_columns_to_predict_rows(all_tables_df, predict_rows, ['manufacturer_name', 'manufacture_country', 'vendor_status'])

            if should_filter_out_pending_vendors:
                predict_rows = predict_rows[predict_rows['vendor_status'] != 'pending_vendor']

            ret_df = pd.concat([ret_df, predict_rows.head(max_recommendations)])
        ret_df.to_csv(csv_filename, index=False)

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
        if self._is_model_trained is True and self._fit_predict_columns is not None:
            missing_columns = list(set(self._fit_predict_columns).difference(set(prepared_data.columns)))
            prepared_data = pd.concat([prepared_data, pd.DataFrame(index=prepared_data.index, columns=missing_columns)],
                                      axis=1)
            prepared_data = prepared_data.fillna(0)

            # Reorder (and filter) columns to the order at train time
            prepared_data = prepared_data[self._fit_predict_columns]
        return prepared_data

    # Feature validation
    # Only necessary features
    # One hot encoding
    def prepare_for_fit_predict(self, raw_data_df, verbose=False):
        # Feature validation
        if self.do_all_features_exist(raw_data_df.columns):
            ret_df = raw_data_df.copy()
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
