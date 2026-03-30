import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)                    # Extract directory path
        os.makedirs(dir_path, exist_ok=True)                     # Create directory if not exists
        with open(file_path, "wb") as file_obj:                  # Open file in write-binary mode
            pickle.dump(obj, file_obj)                           # Serialize and save object
    except Exception as e:
        raise CustomException(e, sys)                            # Raise custom exception on error
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):  
    '''
    * models: Dictionary of model objects → e.g., {'Random Forest': RandomForestRegressor(), 'XGBoost': XGBRegressor()}
    * param: Dictionary of hyperparameters → e.g., {'Random Forest': {'n_estimators': [50, 100]}, 'XGBoost': {...}}
    '''
    try:
        report = {}  # Will store {model_name: test_r2_score}

        for i in range(len(list(models))):
            model = list(models.values())[i]           # Get the model object
            para = param[list(models.keys())[i]]       # Get its hyperparameters

            gs = GridSearchCV(model, para, cv=3)  # 3-fold cross-validation
            gs.fit(X_train, y_train)              # Train with all parameter combinations

            model.set_params(**gs.best_params_)  # Apply the best parameters found
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)