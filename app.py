# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:43:26 2020

@author: paras
"""

from flask import Flask, request
import flask

from sklearn.externals import joblib 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from keras_tqdm import TQDMNotebookCallback
from tqdm import tqdm

import statsmodels
import statsmodels.api as sm

import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import normaltest
from scipy.stats import norm ,rankdata
from scipy.special import boxcox1p
from scipy.optimize import curve_fit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.externals import joblib 

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import plot_importance


app = Flask(__name__)

#///////////////////////////////////////////////////////////////////////

def find_bmi(data):
    bmi = data['weight']/((data['height']/100)**2)
    return bmi

def bp_level(data):
    if (data['ap_hi'] <= 120) and (data['ap_lo'] <= 80):
        return 'normal'
    if (data['ap_hi'] >= 120 and data['ap_hi'] < 129) and (data['ap_lo'] < 80):
        return 'above_normal'
    if (data['ap_hi'] >= 129 and data['ap_hi'] < 139) | (data['ap_lo'] >= 80 and data['ap_lo'] < 89):
        return 'high'
    if (data['ap_hi'] >= 139) | (data['ap_lo'] >= 89):
        return 'very_high'
    if (data['ap_hi'] >= 180) | (data['ap_lo'] >= 120):
        return 'extreme_high'
    
def age_level(data):
    if data["age"] < 30:
        return '0'
    if data['age'] >= 30 and data['age'] < 40:
        return '1'
    if data['age'] >= 40 and data['age'] < 45:
        return '2'
    if data['age'] >= 45 and data['age'] < 50:
        return '3'
    if data['age'] >= 50 and data['age'] < 55:
        return '4'
    if data['age'] >= 55 and data['age'] < 60:
        return '5'
    if data['age'] >= 60 and data['age'] < 70:
        return '6'
    else:
        return '7'
    
def bmi_level(data):
    if data['bmi'] <= 18.5:
        return 'underweight'
    if data['bmi'] > 18.5 and data['bmi'] <= 24.9:
        return 'normal'
    if data['bmi'] > 24.9 and data['bmi'] <= 29.9:
        return 'overweight'
    if data['bmi'] >= 29.9:
        return 'obese'
#///////////////////////////////////////////////////////////////////////

    
#///////////////////////////////////////////////////////////////////////
    
#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////
def clean_data(raw_df):    
    raw_df['age'] = round(raw_df['age']/365.25).apply(lambda x: int(x))
    raw_df['gender']= raw_df['gender'].apply(lambda x: 0 if x==2 else 1)
    raw_df = raw_df[raw_df['ap_hi'] > raw_df['ap_lo']].reset_index(drop=True)
                                         
    raw_df['bmi'] = raw_df.apply(find_bmi, axis=1, result_type='reduce')
    raw_df['bp_level'] = raw_df.apply(bp_level, axis=1, result_type='reduce')
    raw_df['age_level'] = raw_df.apply(age_level, axis=1, result_type='reduce')
    raw_df['bmi_level'] = raw_df.apply(bmi_level, axis=1, result_type='reduce')
    print("DROP COLUMNS: ", drop_columns)
    
    try:
        raw_df.drop(drop_columns, axis=1, inplace=True)
    except:
        print("Custom Error: droping columns not executed!!")
    
    return raw_df
#///////////////////////////////////////////////////////////////////////
    
#///////////////////////////////////////////////////////////////////////
def predict_value(dff):
    #clf = load_model('classifier_model.h5')
    pred = clf.predict(dff) 
    print("FINAL PREDICTION VALUE: ")
    print(pred)
    return pred

    
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict-single', methods=['POST'])
def predict_individual():
    form_data = request.form.to_dict()
    print("FORM DATA")
    print(form_data)
    
    df_input = pd.DataFrame.from_records([form_data])
    df_input = pd.DataFrame(df_input)
    print("INPUT DATAFRAME")
    print(df_input)
    df_input = df_input.astype('int')
    print(df_input.info())
    
    clean_df = clean_data(df_input)
    print("CLEAN DATAFRAME")
    print(clean_df)  
    
    clean_df = pd.get_dummies(clean_df,drop_first=False)
    print(clean_df.info())

    sample_df = pd.DataFrame(columns = main_cols)
    main_df = sample_df.append(clean_df)
    main_df = main_df.fillna(0)
    print("MAIN DATAFRAME")
    print(main_df)
    print(main_df.info())
    print()


    final_df = scaler.transform(main_df)
    print("FINAL DATAFRAME")
    print(final_df)
        
    pred = predict_value(final_df)[0]
    if pred == 1:
        pred_val = "Ouch!, You have a high chance of having a Cardiovascular Disease. Please contact a Cardiologist as soon as possible!"
    else:
        pred_val = "Wohoo! You are perfectly fine!!"
        
    return flask.render_template('index.html', predicted_value="{}".format(str(pred_val)))


# @app.route('/predict-multiple', methods=['POST'])
# def predict_multiple():
#     form_data = request.form.to_dict()
#     print("FORM DATA")
#     form_data_array = np.array(form_data["myarray"])
#     print(form_data_array)

#     js_df = pd.read_json(form_data["myarray"])
    
#     df_input = pd.DataFrame.from_records(js_df)
#     df_input.columns = df_input.iloc[0]
#     df_input = df_input.iloc[1:, :]
#     print("INPUT DATAFRAME")
#     print(df_input)
    
#     clean_df = clean_data(df_input)
#     print("CLEAN DATAFRAME")
#     print(clean_df)   
    
#     sample_df = pd.DataFrame(columns = main_cols)
#     main_df = sample_df.append(clean_df)
#     main_df = main_df.fillna(0)
#     print("MAIN DATAFRAME")
#     print(main_df)
#     main_df = pd.to_numeric(main_df)
#     print(main_df.info())
#     print()
    
#     final_df = standardize_data(main_df)
#     print("FINAL DATAFRAME")
#     print(final_df)
        
#     pred = predict_value(final_df)
    
#     res = pd.DataFrame({"id": df_input["id"], "prediction": pred})
#     print("RESULT")
#     print(res)
    
#     res_json = res.to_json(orient='records')

#     return res_json
#     #return flask.render_template('index.html', predicted_value="Diagnosis report: {}".format(str(pred)))



if __name__ == '__main__':
    main_cols = joblib.load("data_columns.pkl")
    clf = joblib.load("classifier_model.pkl")
    scaler = joblib.load("std_scaler.pkl")
    drop_columns = joblib.load("drop_columns.pkl")
    drop_columns.remove('id')
    
    app.run(host='0.0.0.0', port=8080)