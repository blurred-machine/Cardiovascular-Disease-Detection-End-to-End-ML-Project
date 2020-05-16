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

from sklearn.externals import joblib 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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


#///////////////////////////////////////////////////////////////////////
raw_df = pd.read_csv("cardio_train.csv", sep=";")
#///////////////////////////////////////////////////////////////////////
z = np.abs(stats.zscore(raw_df))
raw_df = raw_df[(z < 3).all(axis=1)]
#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////
df_X = raw_df.drop(["id", "cardio"], axis=1)
df_y = raw_df.loc[:, "cardio"]
df_X = df_X.astype("float")
df_X = df_X.reindex(sorted(df_X.columns), axis=1)

joblib.dump(df_X.columns, 'data_columns.pkl')
 
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
print(X_train.head())
#///////////////////////////////////////////////////////////////////////
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("X_train")
print(X_train)
print("X_test")
print(X_test)

joblib.dump(scaler, 'std_scaler.pkl') 
#/////////////////////////////////////////////////////////////////////
def model_evaluation(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm, end="\n\n")

    TN = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TP = cm[1, 1]

    P = FN+TP
    N = TN+FP

    TPR = TP/P
    TNR = TN/N
    FPR = FP/N
    FNR = FN/P

    accuracy = (TN+TP)/(P+N)
    print("Test Accuracy: "+str(accuracy), end="\n\n")
    print("All 4 parameters: ",TN, FN, FP, TP, end="\n\n")
    print("TPR: {}".format(TPR))
    print("TNR: {}".format(TNR))
    print("FPR: {}".format(FPR))
    print("FNR: {}".format(FNR))
    print()
    
    precision = TP/(TP+FP)
    recall = TPR
    
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print()

    f1 = 2 * (precision * recall) / (precision + recall)
    print("F1-Score (micro): ", f1, end="\n\n")
#/////////////////////////////////////////////////////////////////////
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
print("Training: "+str(xgb_model.score(X_train, y_train)))
xgb_pred = xgb_model.predict(X_test)
model_evaluation(y_test, xgb_pred)
#/////////////////////////////////////////////////////////////////////
# classifier.save('classifier_model.h5')
joblib.dump(xgb_model, 'classifier_model.pkl') 



