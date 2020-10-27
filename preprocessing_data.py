import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import mlflow

import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def preprocessing_train():

    train = pd.read_csv('application_train.csv')
    test = pd.read_csv('application_test.csv')

    ###READING EVERY FEATURE WITH OVER 30% MISSING VALUES

    train_missing = (train.isnull().sum() / len(train)).sort_values(ascending = False)
    test_missing = (test.isnull().sum() / len(test)).sort_values(ascending = False)
    train_missing = train_missing.index[train_missing > 0.3]
    test_missing = test_missing.index[test_missing > 0.3]

    ###DROPPING EVERY FEATURE WITH OVER 30% MISSING VALUES

    for i in train_missing:
        train = train.drop([i], axis = 'columns')

    for i in test_missing:
        test = test.drop([i], axis = 'columns')

    cat_vars = train.select_dtypes('object').columns.tolist()  

    ###FILLING NaN IN NUMERICAL VALUES WITH THE MEAN OF EACH COLUMN

    for col in train:
        if col not in cat_vars:
            train[col] = train[col].fillna(train[col].mean())

    for col in test:
        if col not in cat_vars:
            test[col] = test[col].fillna(test[col].mean())

    ###DROPPING REMAINING LINES WITH NaN VALUES (FROM CATEGORICAL VARIABLES)    
            
    train = train.dropna(axis=0, how='any', subset=cat_vars)
    test = test.dropna(axis=0, how='any', subset=cat_vars)

    ###LABEL ENCODING

    le = LabelEncoder()

    for col in train:
        if train[col].dtype == 'object':
    # If 2 or fewer unique categories
            if len(list(train[col].unique())) <= 2:
                le.fit(train[col])
                train[col] = le.transform(train[col])
                test[col] = le.transform(test[col])

    ###ONE HOT ENCODING

    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    labels = train['TARGET']

    ###ALIGN TRAIN AND TEST

    train, test = train.align(test, join='inner', axis=1)
    train['TARGET'] = labels

    return train

def preprocessing_test():
    
    train = pd.read_csv('application_train.csv')
    test = pd.read_csv('application_test.csv')

    ###READING EVERY FEATURE WITH OVER 30% MISSING VALUES

    train_missing = (train.isnull().sum() / len(train)).sort_values(ascending = False)
    test_missing = (test.isnull().sum() / len(test)).sort_values(ascending = False)
    train_missing = train_missing.index[train_missing > 0.3]
    test_missing = test_missing.index[test_missing > 0.3]

    ###DROPPING EVERY FEATURE WITH OVER 30% MISSING VALUES

    for i in train_missing:
        train = train.drop([i], axis = 'columns')

    for i in test_missing:
        test = test.drop([i], axis = 'columns')

    cat_vars = train.select_dtypes('object').columns.tolist()  

    ###FILLING NaN IN NUMERICAL VALUES WITH THE MEAN OF EACH COLUMN

    for col in train:
        if col not in cat_vars:
            train[col] = train[col].fillna(train[col].mean())

    for col in test:
        if col not in cat_vars:
            test[col] = test[col].fillna(test[col].mean())

    ###DROPPING REMAINING LINES WITH NaN VALUES (FROM CATEGORICAL VARIABLES)    
            
    train = train.dropna(axis=0, how='any', subset=cat_vars)
    test = test.dropna(axis=0, how='any', subset=cat_vars)

    ###LABEL ENCODING

    le = LabelEncoder()

    for col in train:
        if train[col].dtype == 'object':
    # If 2 or fewer unique categories
            if len(list(train[col].unique())) <= 2:
                le.fit(train[col])
                train[col] = le.transform(train[col])
                test[col] = le.transform(test[col])

    ###ONE HOT ENCODING

    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    labels = train['TARGET']

    ###ALIGN TRAIN AND TEST

    train, test = train.align(test, join='inner', axis=1)
    train['TARGET'] = labels

    return test