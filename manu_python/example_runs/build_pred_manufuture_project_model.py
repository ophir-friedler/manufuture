# Follows notebooks/Modelling.ipynb

from manu_python import manu_main
from manu_python.config.config import MANUFACTURER_BID_LABEL_COLUMN_NAME
from manu_python.model_evaluation import evaluators
from manu_python.models import pred_manufacturer_project_bid_submission

# recommendations for one project
MAX_RECOMMENDATIONS = 20

MODEL_NAME = 'num_parts_and_total_quantity'

WRITE_TO_CSV = False

DEPLOY_TO_MODEL_VAR_A = False
DEPLOY_TO_MODEL_VAR_B = False

all_tables_df = manu_main.get_all_tables_df()
bidSubmissionPredictor = pred_manufacturer_project_bid_submission.BidSubmissionPredictor()
bidSubmissionPredictor.build_model(all_tables_df=all_tables_df,
                                   verbose=True
                                   )

print(bidSubmissionPredictor._model)

# display(bidSubmissionPredictor.rank_manufacturers_for_project(all_tables_df, project_id=28719))
print("number of manufacturers: " + str(all_tables_df[bidSubmissionPredictor._input_table_name]['post_id_manuf'].nunique()))
print("number of projects: " + str(all_tables_df[bidSubmissionPredictor._input_table_name]['post_id_project'].nunique()))

# Evaluation:
predictor = bidSubmissionPredictor
prediction_colname = 'predBidProb'
outcome_colname = MANUFACTURER_BID_LABEL_COLUMN_NAME
num_top_manufacturers = 5
evaluation_df = evaluators.evaluate_manufacturers_bid_for_project_ranking(predictor=predictor,
                                                                          all_tables_df=all_tables_df,
                                                                          prediction_colname=prediction_colname,
                                                                          outcome_colname=outcome_colname,
                                                                          num_top_manufacturers=num_top_manufacturers)
