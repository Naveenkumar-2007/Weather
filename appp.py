
from fastapi import FastAPI,Body
import numpy as np
import pickle 
from pydantic import BaseModel
app=FastAPI()

with open('artifacts/model.pkl','rb') as f:
    model_pik=pickle.load(f)
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
    Sunshine:int

@app.post('/predict/')
def prediction_of(item:Base_model):
    rain_today_binary=1 if item.RainToday.lower()=='yes' else 0
    predict=np.array([[item.MinTemp,item.MaxTemp,item.Rainfall,item.Evaporation,item.Humidity9am,item.Humidity3pm,
                      item.Pressure9am,item.Pressure3pm,item.Cloud9am,item.Cloud3pm,item.Temp9am,item.Temp3pm,
                      item.RISK_MM,rain_today_binary,item.Sunshine]])
    test_predict=model_pik.predict(predict)
    return {'prediction:':test_predict[0]}
    
