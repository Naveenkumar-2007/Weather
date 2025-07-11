
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transform import datatransformcong
from src.components.data_transform import datatransform
from src.components.model_tranier import modeltraniercong
from src.components.model_tranier import model_transform
@dataclass
class dataingestioncong:
    traindata:str=os.path.join('artifacts','train.csv')
    testdata:str=os.path.join('artifacts','test.csv')
    rawdata:str=os.path.join('artifacts','raw.csv')

class dataingestion:
    def __init__(self):
        self.data_ingestion=dataingestioncong()

    def get_data_ingestion(self):
        try:
            logging.info('training and testing')
            df=pd.read_csv('C:\\Users\\navee\\Cisco Packet Tracer 8.2.2\\saves\\bentoml\\notebook\\new_weather.csv')
            os.makedirs(os.path.dirname(self.data_ingestion.traindata),exist_ok=True)
            df.to_csv(self.data_ingestion.rawdata,index=False,header=True)
            logging.info('this is covert raw data')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion.traindata,index=False,header=True)

            test_set.to_csv(self.data_ingestion.testdata,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.data_ingestion.traindata,
                self.data_ingestion.testdata
         )
        except Exception as ex:
           raise CustomException(ex,sys)
        
if __name__=='__main__':
   data=dataingestion()
   train_data,test_data=data.get_data_ingestion()
   transform=datatransform()
   train_trans,test_trans,_=transform.get_preprocessing(train_data,test_data)
   model_data=model_transform()
   print(model_data.get_models(train_trans,test_trans))
