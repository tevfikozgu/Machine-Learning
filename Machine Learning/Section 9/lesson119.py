#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 2 13:01:24 2019

@author: tevfikozgu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('Churn_Modelling.csv')

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
le2 = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

import keras
from keras.models import Sequential #yapay sinir agi kullanicaz demek bu
from keras.layers import Dense #objeler burda

classifier = Sequential() #sequentialda tanimlanmis bir yapay sinir agi olusturduk
classifier.add(Dense(6, init='uniform', activation='relu', input_dim=11)) #6 burdaki gizli katman sayisi biz yontem olarak giris ve cikistakilar toplayip 2 ye bolduk ama sa ma aslinda
