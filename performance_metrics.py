import pandas as pd
import numpy as np


def get_perform_metrics_from_log(dataset, outcome_col, prediction_col):
    """
    returns performance metrics for logged binary classification, namely
    accuracy, balanced accuracy, precision, recall, f1-score
     Parameters
     ----------
        dataset : Pandas dataframe with predicted and true binary label
        outcome_col : String that indicates column containing true label
        prediction_col : String that indicates column containing predicted label
     Returns
     ----------
     perform_metrics : list containing the 5 performance metrics
     """
    tp = np.sum(dataset[prediction_col] + dataset[outcome_col] == 2)
    tn = np.sum(dataset[prediction_col] + dataset[outcome_col] == 0)
    fp = np.sum(dataset[prediction_col] > dataset[outcome_col])
    fn = np.sum(dataset[prediction_col] < dataset[outcome_col])

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    true_neg = tn / (tn + fp)
    balanced_acc = (recall + true_neg) / 2
    f1_score = (2 * recall * precision) / (recall + precision)

    perf_metrics = {"accuracy": accuracy,
                    "balanced_acc": balanced_acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score}

    return perf_metrics


def get_fairness_metrics_from_log(dataset, outcome_col, prediction_col, sensitive_attr):
    """
    returns fairness metrics for logged binary classification, namely
    statistical parity difference and average equalized odds difference
     Parameters
     ----------
        dataset : Pandas dataframe with predicted and true binary label
        outcome_col : String that indicates column containing true label
        prediction_col : String that indicates column containing predicted label
        sensitive_attr : Strings indicating binary sensitive attribute
     Returns
     ----------
     fairness_metrics : list containing the 5 performance metrics
     """
    protected_value = dataset[sensitive_attr].value_counts().idxmin()
    my_prot_grp = {sensitive_attr: protected_value}
    my_prot_grp_bool = dataset['age'] == my_prot_grp['age']
    in_group = dataset[my_prot_grp_bool]
    out_group = dataset[~my_prot_grp_bool]

    # compute statistical parity difference
    stat_par_diff = in_group[prediction_col].mean() - out_group[prediction_col].mean()

    # compute average absolute odds difference
    fp_group = np.sum(in_group[outcome_col] > in_group[prediction_col])
    tn_group = np.sum(in_group[outcome_col] + in_group[prediction_col] == 0)
    fn_group = np.sum(in_group[outcome_col] < in_group[prediction_col])
    tp_group = np.sum(in_group[outcome_col] + in_group[prediction_col] == 2)

    fp_rest = np.sum(out_group[outcome_col] > out_group[prediction_col])
    tn_rest = np.sum(out_group[outcome_col] + out_group[prediction_col] == 0)
    fn_rest = np.sum(out_group[outcome_col] < out_group[prediction_col])
    tp_rest = np.sum(out_group[outcome_col] + out_group[prediction_col] == 2)

    neg_group = fp_group + tn_group
    fp_rate_group = fp_group / neg_group  # false positive rate group

    neg_rest = fp_rest + tn_rest
    fp_rate_rest = fp_rest / neg_rest  # false positive rate rest

    pos_group = tp_group + fn_group
    tp_rate_group = tp_group / pos_group  # true positive rate group

    pos_rest = tp_rest + fn_rest
    tp_rate_rest = tp_rest / pos_rest  # true positive rate rest

    avg_abs_odds_diff = 0.5 * (abs(fp_rate_group - fp_rate_rest) + abs(tp_rate_rest - tp_rate_group))

    fairness_metrics = {"statistical_parity_difference": stat_par_diff,
                        "avg_absolute_odds_difference": avg_abs_odds_diff}

    return fairness_metrics
