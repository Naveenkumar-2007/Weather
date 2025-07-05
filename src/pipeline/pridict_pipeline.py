
import os 
import sys
from src.utlies import load_obj
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as ps

@dataclass
class predict:
    def __init__(self):
        pass

    def get_predict(self,feature):
        try:
            model=os.path.join('artifacts','model.pkl')
            preprocessing=os.path.join('artifacts','preprocessing.pkl')
            model_load=load_obj(file_path=model)
            pre_load=load_obj(file_path=preprocessing)
            data=pre_load.transform(feature)
            model_data=model_load.predict(data)

            return model_data
        except Exception as ex:
            raise CustomException(ex,sys)
        
class customdata:
    def __init__(self,
        MinTemp:int,
        MaxTemp:int,
        Rainfall:int,
        Evaporation:int,
        Humidity9am:int,
        Humidity3pm:int,
        Pressure9am:int,
        Pressure3pm:int,
        Cloud9am:int,
        Cloud3pm:int, 
        Temp9am:int, 
        Temp3pm:int, 
        RISK_MM:int,
        RainToday:str,):
        self.MinTemp=MinTemp
        self.MaxTemp=MaxTemp
        self.Rainfall=Rainfall
        self.Evaporation=Evaporation
        self.Humidity9am=Humidity9am
        self.Humidity3pm=Humidity3pm
        self.Pressure9am=Pressure9am
        self.Pressure3pm=Pressure3pm
        self.Cloud9am=Cloud9am
        self.Cloud3pm=Cloud3pm
        self.Temp9am=Temp9am
        self.Temp3pm=Temp3pm
        self.RISK_MM=RISK_MM
        self.RainToday=RainToday

    def data_frame(self):
        try:
            custom_data_frame={
                "MinTemp": [self.MinTemp],
                "MaxTemp":[self.MaxTemp],
                'Rainfall':[self.Rainfall],
                'Evaporation':[self.Evaporation],
                'Humidity9am':[self.Humidity9am],
                'Humidity3pm':[self.Humidity3pm],
                'Pressure9am':[self.Pressure9am],
                'Pressure3pm':[self.Pressure3pm],
                'Cloud9am':[self.Cloud9am],
                'Cloud3pm':[self.Cloud3pm],
                'Temp9am':[self.Temp9am],
                'Temp3pm':[self.Temp3pm],
                'RISK_MM':[self.RISK_MM],
                'RainToday':[self.RainToday],
            }
            return ps.DataFrame(custom_data_frame)
        except Exception as ex:
            raise CustomException(ex,sys)


