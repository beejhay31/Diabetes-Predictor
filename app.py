import flask
from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy as np
from flask_ngrok import run_with_ngrok
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
run_with_ngrok(app)

model = load_model('diabetes_model.pkl')

@app.route('/', methods=['GET'])
def home():
  return render_template('index1.html')

@app.route('/predict', methods=['GET',"POST"])
def predict():
  input_values = [x for x in request.form.values()]
  inp_features = np.array(input_values)
  col = ['AGE','FASTING BLOOD SUGAR (FBS)', 'BODY MASS INDEX (BMI)', 'WAIST', 
         'GENDER', 'FAMILY HISTORY OF DIABETES (FHD)',
       'FAMILY HISTORY OF HYPERTENSION (FHH)', 'HISTORY OF EXCESS URINE (HEU)',
       'PREVIOUS HISTORY OF DIABETES OF ANY TYPE (PHD)']
  df=pd.DataFrame([inp_features], columns=col)
  print(input_values)
  prediction = predict_model(model, data=df)
  prediction = prediction.Label[0]
  if str(prediction)=="Y":
    return render_template('index1.html', prediction_text='You are diabetic')
  else:
    return render_template('index1.html', prediction_text='Congratulations, You are not diabetic')

app.run()
