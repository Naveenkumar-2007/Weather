
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utlies import save_obj

@dataclass
class datatransformcong:
    tranform_file_path=os.path.join('artifacts','preprocessing.pkl')

class datatransform:
    def __init__(self):
        self.data_transform=datatransformcong()
    def get_data_transform(self):
        try:
            num_col=[
                'MinTemp',
                'MaxTemp', 
                'Rainfall', 
                'Evaporation', 
                'Humidity9am', 
                'Humidity3pm', 
                'Pressure9am', 
                'Pressure3pm', 
                'Cloud9am', 
                'Cloud3pm', 
                'Temp9am', 
                'Temp3pm', 
                'RISK_MM',
            ]
            cat_col=['RainToday']
            num_pipeline=Pipeline(
                steps=[('simpleimputer',SimpleImputer(strategy='median')),
                       ('stand',StandardScaler())
                       ]
            )
            cat_col_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            preprocessing=ColumnTransformer(
                [
                    ('num_col',num_pipeline,num_col),
                    ('cat_col',cat_col_pipeline,cat_col)
                ]
            )
            return preprocessing
        except Exception as ex:
            raise CustomException(ex,sys)
    def get_preprocessing(self,trainpath,testpath):
        try:
            logging.info('train and test')
            train_file=pd.read_csv(trainpath)
            test_file=pd.read_csv(testpath)
            logging.info('preprocessing data')
            preprocessing_file_path=self.get_data_transform()
            target_column=['RainTomorrow']
            input_feature_train=train_file.drop(columns=['RainTomorrow'])
            target_train=train_file[target_column]
            input_feature_test=test_file.drop(columns=['RainTomorrow'])
            target_test=test_file[target_column]

            pre_train_path=preprocessing_file_path.fit_transform(input_feature_train)
            pre_test_path=preprocessing_file_path.transform(input_feature_test)

            train_c=np.c_[
                pre_train_path,np.array(target_train)
            ]
            test_c=np.c_[
                pre_test_path,np.array(target_test)
            ]
            save_obj(
                file_path=self.data_transform.tranform_file_path,
                obj=preprocessing_file_path
            )
            return(
                train_c,
                test_c,
                self.data_transform.tranform_file_path,
            )
        except Exception as ex:
            raise CustomException(ex,sys)
