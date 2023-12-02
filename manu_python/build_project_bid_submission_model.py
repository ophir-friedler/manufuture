import logging

import manu_python.manu_main
from manu_python.config.config import MANUFACTURER_BID_LABEL_COLUMN_NAME
from manu_python.model_evaluation import evaluators
from manu_python.models import pred_manufacturer_project_bid_submission

print("Starting to build manufacturer bid submission predictor")
# recommendations for one project
MAX_RECOMMENDATIONS = 20

MODEL_TYPE = 'deep_v0'  # lr, deep_v0
MODEL_NAME = 'num_parts_and_total_quantity'
MODEL_CSV_FILEPATH = '/Users/ofriedler/Dropbox/Work/Consultation/Manufuture/dev/manu_python/model_storage' \
                     '/manufacturers_for_project_' + MODEL_TYPE + '_' + MODEL_NAME + '_' + str(MAX_RECOMMENDATIONS) + \
                     '.csv '
WRITE_TO_CSV = False
EVALUATE = True

DEPLOY_TO_MODEL_VAR_A = False
DEPLOY_TO_MODEL_VAR_B = False

all_tables_df = manu_python.manu_main.get_all_tables_df()

bidSubmissionPredictor = pred_manufacturer_project_bid_submission.BidSubmissionPredictor()
bidSubmissionPredictor.build_model(all_tables_df=all_tables_df,
                                   model_type=MODEL_TYPE,
                                   verbose=True
                                   )

print(bidSubmissionPredictor._model)

if WRITE_TO_CSV:
    logging.info("Writing model to csv: " + MODEL_CSV_FILEPATH)
    bidSubmissionPredictor.rank_for_all_project_features_to_csv(all_tables_df=all_tables_df,
                                                                max_recommendations=MAX_RECOMMENDATIONS,
                                                                csv_filename=MODEL_CSV_FILEPATH)
if EVALUATE:
    print("Evaluating " + str(bidSubmissionPredictor))
    evaluation_df = evaluators.evaluate_manufacturers_bid_for_project_ranking(predictor=bidSubmissionPredictor,
                                                                          all_tables_df=all_tables_df,
                                                                          prediction_colname='predBidProb',
                                                                          outcome_colname='is_manuf_bid',
                                                                          num_top_manufacturers=5)
print("Done. ")
