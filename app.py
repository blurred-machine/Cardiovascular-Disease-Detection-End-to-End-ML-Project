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
    new_df = df.astype("float")
    return new_df



def standardize_data(df):        
    new_df = scaler.transform(df)
    return new_df


def predict_value(dff):
    #clf = load_model('classifier_model.h5')

    pred = clf.predict(dff) 
    print("MAIN_PRED : "+str(pred))
    for i in range(len(pred)):
        if pred[i] >= 0.5:
            prediction = 1
        else:
            prediction = 0
    return [pred, prediction]

    
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
        
    pred = predict_value(final_df)
    pred_percent = pred[0]
    pred_value = pred[1]
        
    return flask.render_template('index.html', predicted_value="Diagnosis report: {} with percent: {}%".format(str(pred_value), str(pred_percent)))

if __name__ == '__main__':
    main_cols = joblib.load("data_columns.pkl")
    clf = joblib.load("classifier_model.pkl")
    scaler = joblib.load("std_scaler.pkl")

    app.run(host='0.0.0.0', port=8080)