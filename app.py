from flask import Flask,request,render_template
import pandas as pd
from src.pipeline.pridict_pipeline import predict,customdata
application=Flask(__name__)
app=application
@app.route('/')
def index():
    return render_template('h.html')

@app.route('/last',methods=['GET','POST'])
def last_predict():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data=customdata(
            MinTemp=float(request.form.get('MinTemp')),
            MaxTemp=float(request.form.get('MaxTemp')),
            Rainfall=request.form.get('Rainfall'),
            Evaporation=request.form.get('Evaporation'),
            Humidity9am=float(request.form.get('Humidity9am')),
            Humidity3pm=float(request.form.get('Humidity3pm')),
            Pressure9am=float(request.form.get('Pressure9am')),
            Pressure3pm=float(request.form.get('Pressure3pm')),
            Cloud9am=float(request.form.get('Cloud9am')),
            Cloud3pm=float(request.form.get('Cloud3pm')), 
            Temp9am=float(request.form.get('Temp9am')), 
            Temp3pm=float(request.form.get('Temp3pm')), 
            RISK_MM=float(request.form.get('RISK_MM')),
            RainToday=request.form.get('RainToday')
        )
        data_df=data.data_frame()
        print(data_df)
        predict_pipeline=predict()
        results=predict_pipeline.get_predict(data_df)
        return render_template('home.html',results=results[0])

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)