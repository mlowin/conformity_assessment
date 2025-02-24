# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:09:51 2023

@author: janmo
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import Explorative_data_analysis.Functions as func
import seaborn as sns
import matplotlib.pyplot as plt     
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score, r2_score
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from functools import reduce
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import shap
from imblearn.over_sampling import SMOTE
from statsmodels.discrete.discrete_model import Logit
import os
import pandas as pd
import pyreadstat 
from scipy import stats
import pickle


Data = pd.read_csv("datasets/Fraud/application_data.csv")
Data.head()
Data.columns
#Data_NaN = (Data.isna().sum()/len(Data))*100

# Sample the dataset
Data = Data.sample(frac=0.5).reset_index(drop=True)

# Fill nas
Data = Data.fillna(-1)

for i in Data.columns:
    if Data[i].dtype == "object":
        print(i, Data[i].nunique())

# Analysze Gender
#Data["CODE_GENDER"].value_counts()

# Drop cats with too many values
Data.drop(["ORGANIZATION_TYPE", "OCCUPATION_TYPE"], axis=1, inplace=True)

# Onehot
Data_onehot = pd.get_dummies(Data, drop_first = True)

# Assign X and y
X = Data_onehot.drop(['TARGET'], axis=1)
y = Data_onehot['TARGET']

# Investigate imbalanceness
sns.countplot(y)
y.value_counts()/len(y)

# Initialize model
#model = XGBClassifier()

# Assign test- and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)


### Oversampling
smote               = SMOTE(sampling_strategy='minority')
X_train_smote, y_train_smote    = smote.fit_resample(X_train, y_train)

### Downsampling


"""
# Parameter tuning
param_grid = [
    {'learning_rate': [0.001],
     #'n_estimators':[500, 1000],
     'max_depth':[3 ,6, 9]
     }
  ]

grid_search = GridSearchCV(model, param_grid,
                           scoring='roc_auc',
                           return_train_score=True,
                           verbose=0)
grid_search.fit(X_train, y_train)
    
final_model = grid_search.best_estimator_
"""



# fit model
final_model = XGBClassifier(learning_rate=0.001, max_depth=3).fit(X_train_smote, y_train_smote)
#final_model = LogisticRegression().fit(X_train, y_train)
#rfb = RandomForestClassifier(n_estimators=100, random_state=13, class_weight="balanced").fit(X_train, y_train)



# Make predictions
pred = final_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels([0, 1]); ax.yaxis.set_ticklabels([0, 1]);

# Performance
accuracy = accuracy_score(y_test, pred)
#balanced_accuracy = balanced_accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
auc      = roc_auc_score(y_test, pred)

"""
# Visualize dist
ax= plt.subplot()
sns.countplot(y_test)
ax.set_title('True Values')
plt.show()

ax= plt.subplot()
sns.countplot(pred)
ax.set_title('Predicted Values')
plt.show()    
"""

### pickle model
filename = 'models/Jan_ist_ein_krasser_Developer_model_v6.sav'
pickle.dump(final_model, open(filename, 'wb'))
Data_onehot.to_csv("datasets/fraud_v5.csv", index=False)
 

