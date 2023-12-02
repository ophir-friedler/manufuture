import logging
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

from manu_python.config.config import MANUFACTURER_BID_LABEL_COLUMN_NAME
from manu_python.models.pred_manufacturer_project_bid_submission import BidSubmissionPredictor


def all_predictions(training_data, target_feature, bidSubmissionPredictor: BidSubmissionPredictor):
    # labels matrix
    labels_df = training_data[[MANUFACTURER_BID_LABEL_COLUMN_NAME]].reset_index().pivot(index="post_id_manuf", columns="post_id_project",
                                                                    values=MANUFACTURER_BID_LABEL_COLUMN_NAME)

    set_of_manufacturers = set(training_data.index.get_level_values('post_id_manuf'))
    set_of_projects = set(training_data.index.get_level_values('post_id_project'))
    predictions_df = pd.DataFrame(index=set_of_manufacturers)
    predictions_df.index.name = 'post_id_manuf'
    for project_post_id in set_of_projects:
        ret_val = pred_bid_submissions_for_project(bidSubmissionPredictor, training_data, target_feature,
                                                   project_post_id=project_post_id)
        all_manufacturers_predictions_for_project = ret_val[['pred_results']].rename(
            columns={'pred_results': project_post_id}).droplevel(level=0)
        predictions_df = predictions_df.join(
            all_manufacturers_predictions_for_project)
    return labels_df, predictions_df


def pred_bid_submissions_for_project(bidSubmissionPredictor: BidSubmissionPredictor, training_data, target_feature, project_post_id):
    # get input for model from training_data
    pred_data = training_data[training_data.index.get_level_values('post_id_project').isin([project_post_id])].copy()
    if len(pred_data) == 0:
        logging.error("project " + str(project_post_id) + " has no rows in training data to predict on")
        return None
    else:
        # y_pred_proba = model.predict_proba(pred_data.drop(columns=[target_feature]))
        pred_data['pred_results'] = bidSubmissionPredictor.model_predict(pred_data.drop(columns=[target_feature]))
        return pred_data


def display_model_metrics_for_manufacturer(predictions_df, labels_matrix, manufacturer_id_to_view):
    pred_and_label = pd.concat(
        [predictions_df.loc[[manufacturer_id_to_view, ]], labels_matrix.loc[[manufacturer_id_to_view, ]]])
    pred_and_label.index = ['predictions', 'labels']
    pred_and_label = pred_and_label.T
    from matplotlib import pyplot

    pyplot.hist(pred_and_label[pred_and_label['labels'] == 1]['predictions'], alpha=0.5, density=True,
                label='positives')
    pyplot.hist(pred_and_label[pred_and_label['labels'] == 0]['predictions'], alpha=0.5, density=True,
                label='negatives')
    pyplot.legend(loc='upper right')
    pyplot.show()

    fpr, tpr, _ = metrics.roc_curve(y_true=pred_and_label['labels'], y_score=pred_and_label['predictions'])
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)

    prec, recall, _ = metrics.precision_recall_curve(y_true=pred_and_label['labels'],
                                                     probas_pred=pred_and_label['predictions'])
    pr_display = metrics.PrecisionRecallDisplay(precision=prec, recall=recall)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.show()


def evaluate_manufacturers_bid_for_project_ranking(predictor: BidSubmissionPredictor,
                                                   all_tables_df,
                                                   prediction_colname,
                                                   outcome_colname,
                                                   num_top_manufacturers):
    projects_with_bids_df = all_tables_df['pam_project_active_manufacturer_th_4_label_reqs']
    # Do not account projects without bids in the evaluation (same score for all models == noise)
    projects_with_bids_df = projects_with_bids_df[projects_with_bids_df['competing_manufacturers'].apply(len) > 0]
    evaluation_projects_set = set(projects_with_bids_df['post_id_project'])
    evaluation_map = {}
    for project_id in evaluation_projects_set:
        evaluation_map[project_id] = evaluate_project(predictor, all_tables_df, project_id, prediction_colname, outcome_colname, num_top_manufacturers)
    evaluation_map_df = pd.DataFrame.from_dict(evaluation_map, orient='index')
    evaluation_map_df.columns = ['success_pct', 'unique_preds_pct']
    evaluation_map_df.sort_values(by=['success_pct'], ascending=False)
    n, bins, patches = plt.hist(evaluation_map_df['success_pct'], 10, facecolor='blue', alpha=0.5)
    n, bins, patches = plt.hist(evaluation_map_df['unique_preds_pct'], 10, facecolor='red', alpha=0.5)

    print("Avg success: " + str(evaluation_map_df['success_pct'].sum() / len(evaluation_map_df['success_pct'])))
    print("Avg uniqueness: " + str(evaluation_map_df['unique_preds_pct'].sum() / len(evaluation_map_df['unique_preds_pct'])))
    plt.show()
    return evaluation_map_df


# Return:
# First entry in list: % of manufacturers out of num_top_manufacturers that actually bid
# Second entry in list: % of unique predictions (desired value = 100)
def evaluate_project(predictor, all_tables_df, project_id, prediction_colname, outcome_colname, num_top_manufacturers, verbose=False):
    predictions_df = predictor.rank_manufacturers_for_project(all_tables_df, project_id)
    recommended_manufacturers_df = predictions_df.sort_values(by=[prediction_colname], ascending=False).head(num_top_manufacturers)
    num_unique_preds = recommended_manufacturers_df[prediction_colname].nunique()
    if verbose:
        print(recommended_manufacturers_df[['post_id_manuf', 'post_id_project', prediction_colname]])
    num_bids = len(predictions_df['competing_manufacturers'].iloc[0])
    success_pct = -1 if num_bids == 0 else 100.00 * recommended_manufacturers_df[outcome_colname].sum() / min(num_bids, num_top_manufacturers)
    unique_preds_pct = 100.00 * num_unique_preds / num_top_manufacturers
    return [success_pct, unique_preds_pct]
