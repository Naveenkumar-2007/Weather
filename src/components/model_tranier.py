
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utlies import save_obj,evalute_model

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score

@dataclass
class modeltraniercong:
    modeltrainer_path_file=os.path.join('artifacts','model.pkl')
class model_transform:
    def __init__(self):
     self.model_tranier=modeltraniercong()

    def get_models(self,trainarray,testarray):
        try:
           x_train,y_train,x_test,y_test=(
              trainarray[:,:-1],
              trainarray[:,-1],
              testarray[:,:-1],
              testarray[:,-1]
           )
           model={
              "Random Forest": RandomForestClassifier(),
              "Decision Tree": DecisionTreeClassifier(),
              "Gradient Boosting": GradientBoostingClassifier(),
              
              "SVM classifier":SVC(),

           }
           param={
              'Random Forest':{
                 'n_estimators':[1,50,100,150,200],
              },
              'Decision Tree':{
                 'min_samples_split':[1,2,3,4,5],
                 
              },
              'Gradient Boosting':{
                 'learning_rate':[0.1,0.2,0.3,0.4,0.5],
              },
              
              'SVM classifier':{
                 'max_iter':[1,2,3,4,5],
              }
           }
           model_report:dict=evalute_model(X_train=x_train,Y_train=y_train,X_test=x_test,Y_test=y_test,
                                         models=model,params=param)
           best_model_score=max(list(sorted(model_report.values())))
           best_model_name=list(model_report.keys())[
                            list(model_report.values()).index(best_model_score)
                            ]
           best_model=model[best_model_name]
           save_obj(
              file_path=self.model_tranier.modeltrainer_path_file,
              obj=best_model
              
           )
           predict=best_model.predict(x_test)
           acc=accuracy_score(y_test,predict)
           return acc
        except Exception as ex:
           raise CustomException(ex,sys)