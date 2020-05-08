#!/usr/bin/env python
# coding: utf-8
"""
Created on Fri May  8 08:32:32 2020

@author: paras
"""

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


from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb

raw_df = pd.read_csv("cardio_train.csv", sep=";")
print(raw_df.columns)
 
z = np.abs(stats.zscore(raw_df))

Q1 = raw_df.quantile(0.25)
Q3 = raw_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

raw_df[["age", "height", "weight", "ap_hi"]] = raw_df[["age", "height", "weight", "ap_hi"]][(z < 3.5).all(axis=1)]
print(raw_df.shape)
print( raw_df["ap_hi"].shape)
print(raw_df.isna().sum())

raw_df = raw_df.ix[raw_df["age"] > 0]





def rank_based_normalization(x):  
    newX = norm.ppf(rankdata(x)/(len(x) + 1))
    return newX

def feature_engg(feature):
    fig, axes = plt.subplots(1,2, figsize=(21,6))
    sns.distplot(feature, ax=axes[0])
    sm.qqplot(feature, stats.norm, fit=True, line='45', ax=axes[1]);
    k2_1, p1 = normaltest(feature)
    print("Normal test P1: ",p1)

    fig, axes = plt.subplots(1,2, figsize=(21,6))
    sns.distplot(rank_based_normalization(feature), ax=axes[0])
    axes[0].set_xlabel('Normalized')
    sm.qqplot(rank_based_normalization(feature), stats.norm, fit=True, line='45', ax=axes[1]);
    k2_2, p2 = normaltest(rank_based_normalization(feature))
    print("Normal test P2: ",p2)
    return rank_based_normalization(feature)


def generate_report():
    y_pred = classifier.predict(X_test)    
    for i in range(len(y_pred)):
        if y_pred[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1, 1])/(cm[0,0]+cm[1, 1]+cm[1,0]+cm[0, 1])
    print("\nTest Accuracy: "+str(accuracy)+"\n")
    
    

   
raw_df['age'] = feature_engg(raw_df['age'])
raw_df['height'] = feature_engg(raw_df['height'])
raw_df['weight'] = feature_engg(raw_df['weight'])

raw_df['ap_hi'] = feature_engg(raw_df['ap_hi'])
raw_df['ap_lo'] = feature_engg(raw_df['ap_lo'])



cholesterol_encoding = pd.get_dummies(raw_df['cholesterol'], prefix="cholesterol")
gluc_encoding = pd.get_dummies(raw_df['gluc'], prefix="gluc")
gender_encoding = pd.get_dummies(raw_df['gender'], prefix="gender")

raw_df = pd.concat([raw_df, cholesterol_encoding], axis=1)
raw_df = raw_df.drop(["cholesterol"], axis=1)

raw_df = pd.concat([raw_df, gluc_encoding], axis=1)
raw_df = raw_df.drop(["gluc"], axis=1)

raw_df = pd.concat([raw_df, gender_encoding], axis=1)
raw_df = raw_df.drop(["gender"], axis=1)



raw_df.to_csv('clean_df.csv', index = False)
clean_df = pd.read_csv("clean_df.csv")

df_X = clean_df.drop(["id", "cardio"], axis=1)
df_y = clean_df.loc[:, "cardio"]
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train, X_test)

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = df_X.shape[1]))
# classifier.add(Dropout(0.2))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(df_X, df_y, validation_split=0.33, batch_size = 70, epochs = 20)

generate_report()

classifier.summary()

# classifier = load_model('my_model.h5')

# save the model so created above into a picle.
classifier.save('my_model.h5')


