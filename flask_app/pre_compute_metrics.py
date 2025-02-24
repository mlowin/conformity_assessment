import pickle
import compute_metrics as compute_metrics
import pandas as pd

import json
file = open('metrics.pickle', 'rb')
metrics = pickle.load(file)
file.close()

# print(metrics)
with open('metrics_start.json', 'w') as f:
    f.write(json.dumps(metrics))
# df = pd.read_csv('../datasets/Fraud_Germany_2023.csv')
# outcome_col = 'TARGET'
# sensitive_attr = 'CODE_GENDER_M'
# model_path = '../models/Fraud_Classifier_XGBoost.sav'

# metrics = {}


# output = {'fairness': {'stat_par_mean': [], 'stat_par_std': [], 'eq_odds_mean':[], 'eq_odds_std': [], 'eq_opp_mean': [], 'eq_opp_std': []}, 'explainability': {'mean_stability': []}, 'performance': {'f1_mean': [], 'f1_std': [], 'accuracy_mean': [], 'accuracy_std': [], 'balanced_acc_mean': [], 'balanced_acc_std': [], 'precision_mean': [], 'precision_std': [], 'recall_mean': [], 'recall_std': [], 'data_drift': []}}

# for thres in range(1,100):
#     print("=======================",thres,"===============================")
#     m = compute_metrics.compute_metrics(Data=df, model_name=model_path,
#                                         column=outcome_col, sensitive_attr=sensitive_attr, threshold=thres/100)
#     for x in m:
#         for y in m[x]:
#             output[x][y].append(m[x][y])
            
# filehandler = open("metrics.obj","wb")
# pickle.dump(output,filehandler)
# filehandler.close()

# print(output)

