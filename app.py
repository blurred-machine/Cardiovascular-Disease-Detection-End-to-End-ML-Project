# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:43:26 2020

@author: paras
"""

from flask import Flask, jsonify, request
import json
import flask


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle 
from sklearn.externals import joblib 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

import scipy.stats as stats
from scipy.stats import norm ,rankdata

from scipy.special import boxcox1p
from scipy.stats import normaltest
import statsmodels
import statsmodels.api as sm
from scipy.optimize import curve_fit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PowerTransformer



app = Flask(__name__)


def clean_data(df):
    #TODO:
    try:
        df = df.drop(["id"], axis=1)
    except:
        print("Custom Error: 'id' column does not exist")
    
    new_df = df.astype("float")
    return new_df



def standardize_data(df):        
    new_df = scaler.transform(df)
    return new_df


def predict_value(dff):
    #clf = load_model('classifier_model.h5')
    pred = clf.predict(dff) 
    print("FINAL PREDICTION VALUE: ")
    print(pred)
    return pred

    
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_individual():
    form_data = request.form.to_dict()
    print("FORM DATA")
    print(form_data)
    
    df_input = pd.DataFrame.from_records([form_data])
    df_input = df_input.drop(['submitBtn'], axis=1)
    df_input = pd.DataFrame(df_input)
    print("INPUT DATAFRAME")
    print(df_input)
    
    clean_df = clean_data(df_input)
    print("CLEAN DATAFRAME")
    print(clean_df)  

    sample_df = pd.DataFrame(columns = main_cols)
    main_df = sample_df.append(clean_df)
    main_df = main_df.fillna(0)
    print("MAIN DATAFRAME")
    print(main_df)
    print(main_df.info())
    print()
    
    final_df = standardize_data(main_df)
    print("FINAL DATAFRAME")
    print(final_df)
        
    pred = predict_value(final_df)[0]
        
    return flask.render_template('index.html', predicted_value="Diagnosis report: {}".format(str(pred)))


@app.route('/predict_multi', methods=['POST'])
def predict_multiple():
    form_data = request.form.to_dict()
    print("FORM DATA")
    form_data_array = np.array(form_data["myarray"])
    print(form_data_array)

    js_df = pd.read_json(form_data["myarray"])
    
    df_input = pd.DataFrame.from_records(js_df)
    df_input.columns = df_input.iloc[0]
    df_input = df_input.iloc[1:, :]
    print("INPUT DATAFRAME")
    print(df_input)
    
    clean_df = clean_data(df_input)
    print("CLEAN DATAFRAME")
    print(clean_df)   
    
    sample_df = pd.DataFrame(columns = main_cols)
    main_df = sample_df.append(clean_df)
    main_df = main_df.fillna(0)
    print("MAIN DATAFRAME")
    print(main_df)
    print(main_df.info())
    print()
    
    final_df = standardize_data(main_df)
    print("FINAL DATAFRAME")
    print(final_df)
        
    pred = predict_value(final_df)

    return jsonify({'prediction': str(pred)})
    #return flask.render_template('index.html', predicted_value="Diagnosis report: {}".format(str(pred)))



if __name__ == '__main__':
    main_cols = joblib.load("data_columns.pkl")
    clf = joblib.load("classifier_model.pkl")
    scaler = joblib.load("std_scaler.pkl")

    app.run(host='0.0.0.0', port=8080)