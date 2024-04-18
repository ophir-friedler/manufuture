import manu_python.manu_main
from manu_python.models import pred_manufacturer_project_bid_submission

print("Starting to build manufacturer bid submission predictor")
# recommendations for one project
MAX_RECOMMENDATIONS = 20

MODEL_NAME = 'num_parts_and_total_quantity'
WRITE_TO_CSV = False
EVALUATE = True

DEPLOY_TO_MODEL_VAR_A = False
DEPLOY_TO_MODEL_VAR_B = False

all_tables_df = manu_python.manu_main.get_all_tables_df()

bidSubmissionPredictor = pred_manufacturer_project_bid_submission.BidSubmissionPredictor()
bidSubmissionPredictor.build_model(all_tables_df=all_tables_df,
                                   verbose=True
                                   )

print(bidSubmissionPredictor._model)
