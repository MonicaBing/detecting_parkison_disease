#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objectuve: to build a model to accurately detect the presence of Parkinson's disease in an individual

lib: XGBoost, install in conda rather than terminal 

Data: UCI ML Parkinsons dataset, parkinsons.data




"""

import numpy as np 
import pandas as pd # interact with data 
import os, sys
from sklearn.preprocessing import MinMaxScaler 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
 
#read the data
df=pd.read_csv('/Users/temp/Desktop/software/python/parkinson_disease_detection/parkinsons.data')
df.head()


#get the features and labels (status column)
# .loc = locate apart from status columnm, .values = convert them into values, get all rows and columns apart from the first column
features = df.loc[:, df.columns != 'status'].values[:,1:] #training 
labels = df.loc[:,'status'].values #training 


#get the count of 1 and 0 
print(labels[labels==1].shape[0], labels[labels==0].shape[0])

# MinMaxScaler = normalise the features data
#fit_transform only used in training data, fit= find the mean and sd fro each feature -> transform featrues
# usually testing only uses transofrm as it used the mean and sd from the fitted training data 
scaler = MinMaxScaler((-1,1))
x= scaler.fit_transform(features)
y = labels

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state =7)

model = XGBClassifier() 
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
