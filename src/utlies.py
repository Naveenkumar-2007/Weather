
import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_obj(file_path,obj):
    try:
        dir=os.path.dirname(file_path)
        os.makedirs(dir,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as ex:
         raise CustomException(ex,sys)
    
def evalute_model(X_train,Y_train,X_test,Y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(Y_train, y_train_pred)

            test_model_score = accuracy_score(Y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as ex:
        raise CustomException(ex,sys)
    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_load:
            return pickle.load(file_load)
    except Exception as ex:
        raise CustomException(ex,sys)