import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

from preprocessing_data import preprocessing_train, preprocessing_test

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    train = preprocessing_train()

    X = train.drop(columns=['TARGET'])
    y = train['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    ne = int(sys.argv[2]) if len(sys.argv) > 500 else 100
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 1
 
    with mlflow.start_run():

        model = XGBClassifier(learning_rate=lr, n_estimators=ne, n_jobs=nj)
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        precision,recall,fscore,support=score(y_test, predicted_qualities)
        precision0 = precision[0]
        precision1 = precision[1]
        recall0 = recall[0]
        recall1 = recall[1]
        fscore0 = fscore[0]
        fscore1 = fscore[1]
        support0 = support[0]
        support1 = support[1]

        accuracy = accuracy_score(y_test, predicted_qualities)

        print("XGB Model (learning_rate=%f, n_estimators=%f, n_jobs=%f):" % (lr, ne, nj))

        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("n_estimators", ne)
        mlflow.log_param("n_jobs", nj)
        mlflow.log_metric("precision0", precision0)
        mlflow.log_metric("precision1", precision1)
        mlflow.log_metric("recall0", recall0)
        mlflow.log_metric("recall1", recall1)
        mlflow.log_metric("fscore0", fscore0)
        mlflow.log_metric("fscore1", fscore1)
        mlflow.log_metric("support0", support0)
        mlflow.log_metric("support1", support1)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "XGBoostClassifier")