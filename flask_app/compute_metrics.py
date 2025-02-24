# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 11:22:49 2022

@author: janmo
"""
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
import itertools
import statistics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score, r2_score, balanced_accuracy_score
import shap
import pickle
import pandas as pd
import scipy
from xgboost import XGBClassifier
import numpy as np
from scipy.stats import ks_2samp

def tanimoto_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def data_drift_test(Data_1, Data_2):

    affected_columns = []

    for col in Data_1.columns:
        
        # Test for data drift using the Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(Data_1[col], Data_2[col])
        
        if p_value < 0.05:
            affected_columns.append(col)
        else:
            pass
        
    number_affected_columns = len(affected_columns)
    perc_affected_columns   = number_affected_columns/len(Data_1.columns)
    
    return number_affected_columns, perc_affected_columns

def compute_metrics(Data, model_name, column, sensitive_attr, threshold=None):
    
    # load pickle model
    model = pickle.load(open(model_name, 'rb'))
    
    # Start K-fold CV
    kf = KFold(n_splits=5) # Define the split
    
    # fairness
    stat_parity_diff_list   = []
    avg_abs_odds_diff_list  = []
    eq_opp_diff_list = []
    
    # explainability
    list_rankings      = []
    list_top_features  = []
    
    # performance
    accuracy_list           = []
    balanced_accuracy_list  = []
    precision_list          = []
    recall_list             = []
    f1_list                 = []
    auc_list                = []
    
    # Assign X and y
    X = Data.drop([column], axis=1)
    y = Data.loc[:, column]
    
    # Iterate over the folds
    for train_index, test_index in kf.split(Data):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        ### Oversampling
        #smote = SMOTE(sampling_strategy='minority')
        #X_train, y_train = smote.fit_resample(X_train, y_train)  
        
        ### fit the XGB model
        #xgb_clf             = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss').fit(X_train_smote, y_train_smote)
        #y_predict_xgb       = xgb_clf.predict(X_test)
        
        # ----------------- Apply Shap ---------------- #
        X_train_shap = X_train.copy(deep=True)
        X_train_shap = X_train_shap.sample(frac=0.1)
        X_test_shap = X_test.copy(deep=True)
        X_test_shap = X_test_shap.sample(frac=0.1)
        explainer           = shap.TreeExplainer(model, X_train_shap)
        shap_values         = explainer.shap_values(X_test_shap, check_additivity=False)
        shap_values_global  = list(abs(shap_values).mean(axis=0)) 
        top_features_idx = set(np.argsort(shap_values_global)[::-1][:10])
        
        list_rankings.append(shap_values_global)
        list_top_features.append(top_features_idx)
        
        # -------------- Apply Performance ------------ #
        if threshold is None:
            pred = model.predict(X_test)
        else:
            pred_proba = model.predict_proba(X_test)
            pred = np.asarray([int(elem[1] > threshold) for elem in pred_proba])
        print('Predicted classes:', pred.sum()/len(pred))
        
        ### compute performance scores
        #auc      = roc_auc_score(y_test, pred)
        f1       = f1_score(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        balanced_accuracy = balanced_accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        
        ### append performance scores
        #auc_list.append(auc)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        balanced_accuracy_list.append(balanced_accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        
        # ------------- Get Fairness Metrics -------- #
        protected_value = X_test[sensitive_attr].value_counts().idxmin()
        my_prot_grp = {sensitive_attr: protected_value}
        my_prot_grp_bool = X_test[sensitive_attr] == my_prot_grp[sensitive_attr]
        
        in_group = X_test[my_prot_grp_bool]
        in_group["outcome"] = y_test[my_prot_grp_bool]
        in_group["pred"] = pred[my_prot_grp_bool]
        
        out_group = X_test[~my_prot_grp_bool]
        out_group["outcome"] = y_test[~my_prot_grp_bool]
        out_group["pred"] = pred[~my_prot_grp_bool]
        
        ### compute statistical parity difference
        stat_par_diff = in_group["pred"].mean() - out_group["pred"].mean()
        stat_parity_diff_list.append(stat_par_diff)
        

        ### compute equal opportunity difference (difference in recall)
        eq_opp_diff = recall_score(in_group["outcome"], in_group["pred"], pos_label=0) - recall_score(out_group["outcome"], out_group["pred"], pos_label=0)
        eq_opp_diff_list.append(eq_opp_diff)


        ### compute average absolute odds difference
        in_group_cm = confusion_matrix(in_group["outcome"], in_group["pred"])
        tn_group, fp_group, fn_group, tp_group = in_group_cm.ravel()
        neg_group = fp_group + tn_group
        pos_group = tp_group + fn_group
        
        out_group_cm = confusion_matrix(out_group["outcome"], out_group["pred"])
        tn_rest, fp_rest, fn_rest, tp_rest = out_group_cm.ravel()
        neg_rest = fp_rest + tn_rest
        pos_rest = tp_rest + fn_rest

        fp_rate_group = fp_group / neg_group  # false positive rate group

        fp_rate_rest = fp_rest / neg_rest  # false positive rate rest

        tp_rate_group = tp_group / pos_group  # true positive rate group

        tp_rate_rest = tp_rest / pos_rest  # true positive rate rest

        avg_abs_odds_diff = 0.5 * (abs(fp_rate_group - fp_rate_rest) + abs(tp_rate_rest - tp_rate_group))

        avg_abs_odds_diff_list.append(avg_abs_odds_diff)
    
    # Create all ranking pairs
    ranks_combinations      = list(itertools.combinations(list_rankings, 2))  
    top_features_combinations = list(itertools.combinations(list_top_features, 2))  

    ### iterate over all combinations and get their Pearson Correlation   
    list_SHAP_correlations       = []
    list_SHAP_similarities       = []
     
    for i in range(len(top_features_combinations)):    
        #corr = scipy.stats.pearsonr(ranks_combinations[i][0], ranks_combinations[i][1])
        #list_SHAP_correlations.append(corr[0])
        tanimoto_sim = tanimoto_similarity(top_features_combinations[i][0], top_features_combinations[i][1])
        list_SHAP_similarities.append(tanimoto_sim)
    
        
    ### get the mean correlation   
    #mean_corr_SHAP   = statistics.mean(list_SHAP_correlations) 
    mean_similarity_SHAP   = statistics.mean(list_SHAP_similarities) 
    #stdev_corr  = statistics.stdev(list_SHAP_correlations)
    
    ### performance
    #mean_AUC = statistics.mean(auc_list)
    #stdv_AUC = statistics.stdev(auc_list)
    
    mean_f1 = statistics.mean(f1_list)
    stdv_f1 = statistics.stdev(f1_list)
    
    mean_accuracy = statistics.mean(accuracy_list)
    stdv_accuracy = statistics.stdev(accuracy_list)
    
    mean_bal_accuracy = statistics.mean(balanced_accuracy_list)
    stdv_bal_accuracy = statistics.stdev(balanced_accuracy_list)
    
    mean_precision = statistics.mean(precision_list)
    stdv_precision = statistics.stdev(precision_list)
    
    mean_recall = statistics.mean(recall_list)
    stdv_recall = statistics.stdev(recall_list)
    
    ### fairness
    mean_stat_par = statistics.mean(stat_parity_diff_list)
    stdv_stat_par = statistics.stdev(stat_parity_diff_list)
    
    mean_eq_odds = statistics.mean(avg_abs_odds_diff_list)
    stdv_eq_odds = statistics.stdev(avg_abs_odds_diff_list)

    mean_eq_opp = statistics.mean(eq_opp_diff_list)
    stdv_eq_opp = statistics.stdev(eq_opp_diff_list)
    
    
    ### -------- Data Drift analysis ---------- ###
    Train_data_initial = pd.read_csv("../datasets/Train_data/Train_data_initial.csv")
    number_drifted_columns, perc_drifted_columns = data_drift_test(Train_data_initial, X_test)
    ### ---------------------------------------- ###
    
    
    stability_summary = {}
    stability_summary['fairness'] = {
                        "stat_par_mean": mean_stat_par,
                        "stat_par_std": stdv_stat_par,
                        
                        "eq_odds_mean": mean_eq_odds,
                        "eq_odds_std": stdv_eq_odds,

                        "eq_opp_mean": mean_eq_opp,
                        "eq_opp_std": stdv_eq_opp,}
    stability_summary['explainability'] = {
                        "mean_stability":mean_similarity_SHAP,
                        }
    stability_summary['performance'] = {
                        #"AUC_mean": mean_AUC,
                        #"AUC_std": stdv_AUC,
                        
                        "f1_mean": mean_f1,
                        "f1_std": stdv_f1,
                        
                        "accuracy_mean":mean_accuracy,
                        "accuracy_std":stdv_accuracy,
                        
                        "balanced_acc_mean": mean_bal_accuracy,
                        "balanced_acc_std": stdv_bal_accuracy,
                        
                        "precision_mean": mean_precision,
                        "precision_std": stdv_precision,
                        
                        "recall_mean":mean_recall,
                        "recall_std":stdv_recall,
                        
                        "data_drift": perc_drifted_columns,
                        }

    return stability_summary


# Data = pd.read_csv("datasets/fraud_v5.csv")
# result = compute_metrics(Data, 'models/Jan_ist_ein_krasser_Developer_model_v6.sav', 'TARGET', 'CODE_GENDER_M')




