
# importing the dependencies 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score
from datetime import datetime
from itertools import cycle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import init_notebook_mode
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask,request,render_template
import pickle 

app=Flask(__name__)

model=pickle.load(open('model3.pkl','rb'))

@app.route('/')
def Home():
    result=''
    return render_template('index.html',**locals())



@app.route('/predict',methods=['POST','GET'])
def predict():
    High=float(request.form['highValue'])
    Low=float(request.form['lowValue'])
    Close=float(request.form['closeValue'])
    result= model.predict([[Close, Low,High]])[0]
    return render_template('index.html',**locals())

if __name__=="__main__":
    app.run(port=3000,debug=True)
