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

from preprocessing_data import preprocessing

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    train = preprocessing()

    X = train.drop(columns=['TARGET'])
    y = train['TARGET']

    # Split the data into training and test sets. (0.75, 0.25) split.
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    # fit model no training data

    # The predicted column is "quality" which is a scalar from [3, 9]

    # Set default values if no alpha is provided
  
    """print('learning_rate: ')
    lr = float(input())
    print('n_estimators: ')
    ne = int(input())
    print('n_jobs: ')
    nj = int(input())"""


    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    ne = int(sys.argv[2]) if len(sys.argv) > 500 else 100
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        model = XGBClassifier(learning_rate=lr, n_estimators=ne, n_jobs=nj)
        model.fit(X_train, y_train)

        # Evaluate Metrics
        predicted_qualities = model.predict(X_test)
        #(rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

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

        # Print out metrics
        print("XGB Model (learning_rate=%f, n_estimators=%f, n_jobs=%f):" % (lr, ne, nj))
        #print("  Precision: %s" % precision)
        #print("  Recall: %s" % recall)
        #print("  Fscore: %s" % fscore)
        #print("  Support: %s" % support)

        # Log parameter, metrics, and model to MLflow
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