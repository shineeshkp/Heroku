#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:43:27 2020

@author: Pratik Barjatiya
"""

import os
import pickle

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbours Classifier model
from sklearn.preprocessing import MinMaxScaler  # Pre-processing method
from sklearn.svm import SVC  # Support Vector Classifier model
from subprocess import check_output

os.getcwd()
os.chdir(os.getcwd() + "/Downloads/backorder_prediction/")


print(check_output(["ls", "input"]).decode("utf8"))

train_df = pd.read_csv("input/Training_Dataset_v2.csv")
train_df.loc[0:4, 'sales_9_month':'potential_issue']
train_df = train_df.drop('sku', axis=1)
train_df = train_df[:-1]

Cols_for_str_to_bool = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
                        'stop_auto_buy', 'rev_stop', 'went_on_backorder']
for col_name in Cols_for_str_to_bool:
    train_df[col_name] = train_df[col_name].map({'No': 0, 'Yes': 1})

train_df.perf_6_month_avg = train_df.perf_6_month_avg.fillna(train_df.perf_6_month_avg.median())
train_df.perf_12_month_avg = train_df.perf_6_month_avg.fillna(train_df.perf_12_month_avg.median())
train_df.lead_time = train_df.lead_time.fillna(train_df.lead_time.median())
dataset = train_df.copy(deep=True)

train_df = train_df.sample(frac=.25)
# Features chosen
features = ['national_inv', 'lead_time', 'sales_1_month', 'pieces_past_due', 'perf_6_month_avg',
            'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']

reduced_train_df = train_df[features]

# Set labels
train_label = train_df['went_on_backorder']

X_train, X_test, y_train, y_test = train_test_split(reduced_train_df, train_label, test_size=0.2, random_state=0)

# Change scale of data
# The label is already in the range 0-1, so it won't be affected by this.
pp_method = MinMaxScaler()
pp_method.fit(X_train)

X_train = pp_method.transform(X_train)
X_train = pd.DataFrame(X_train, columns=features)

X_test = pp_method.transform(X_test)
X_test = pd.DataFrame(X_test, columns=features)

# KNN
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('KNN model score is', score)
# KNN model score is 0.9933880772101952
pickle.dump(model, open('models/knn_model.pkl', 'wb'))
# knn_model = pickle.load(open('models/knn_model.pkl', 'rb'))

# SVC
model = SVC(C=1, kernel='rbf', random_state=10)  # Random state to get a consistent score
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('SVC model score is', score)
# SVC model score is 0.9934710224781677
pickle.dump(model, open('models/svc_model.pkl', 'wb'))
# svc_model = pickle.load(open('models/svc_model.pkl', 'rb'))

# Random Forest
model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2,
                               oob_score=True, random_state=10)  # Random state to get a consistent score
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('Random Forest model score is', score)
# Random Forest model score is 0.9929496522223407
pickle.dump(model, open('models/rf_model.pkl', 'wb'))
# rf_model=pickle.load(open('models/rf_model.pkl', 'rb'))
