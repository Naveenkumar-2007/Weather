
from fastapi import FastAPI,Body
import numpy as np
import pickle 
import pandas as pd
from pydantic import BaseModel
app=FastAPI()

with open('artifacts/model.pkl','rb') as f:
    model_pik=pickle.load(f)
with open('artifacts/preprocessing.pkl','rb') as f:
    pre_pik=pickle.load(f)

class Base_model(BaseModel):
    MinTemp:int
    MaxTemp:int
    Rainfall:int
    Evaporation:int
    Humidity9am:int
    Humidity3pm:int
    Pressure9am:int
    Pressure3pm:int 
    Cloud9am:int
    Cloud3pm:int 
    Temp9am:int 
    Temp3pm:int
    RISK_MM:int
    RainToday:str
    
@app.post('/predict/')
def prediction_of(item:Base_model):
    rain_today_binary=1 if item.RainToday.lower()=='yes' else 0
    input_df = pd.DataFrame([{
        "MinTemp": item.MinTemp,
        "MaxTemp": item.MaxTemp,
        "Rainfall": item.Rainfall,
        "Evaporation": item.Evaporation,
        "Humidity9am": item.Humidity9am,
        "Humidity3pm": item.Humidity3pm,
        "Pressure9am": item.Pressure9am,
        "Pressure3pm": item.Pressure3pm,
        "Cloud9am": item.Cloud9am,
        "Cloud3pm": item.Cloud3pm,
        "Temp9am": item.Temp9am,
        "Temp3pm": item.Temp3pm,
        "RISK_MM": item.RISK_MM,
        
        "RainToday": rain_today_binary # if needed
    }])

    rain_today_binary=1 if item.RainToday.lower()=='yes' else 0
    
    pre=pre_pik.transform(input_df)
    test_predict=model_pik.predict(pre)
    return {'prediction:':test_predict[0]}
    
