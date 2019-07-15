#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 2 13:29:24 2019

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

import keras #bunun altinda tensorflow var aslinda onu da ekledik
from keras.models import Sequential #yapay sinir agi kullanicaz demek bu
from keras.layers import Dense #objeler burda

classifier = Sequential() #sequentialda tanimlanmis bir yapay sinir agi olusturduk
classifier.add(Dense(6, init='uniform', activation='relu', input_dim=11)) #6 burdaki gizli katman sayisi biz yontem olarak giris ve cikistakilar toplayip 2 ye bolduk ama sacma aslinda
classifier.add(Dense(6, init='uniform', activation='relu')) #gizli bir katman daha ekledik
# Iki kere ust uste calistirinca ayni modulleri import ettigi icin hata veriyor. Spyderdan preferencesa gir. Python console gir UMR yi kapat.

classifier.add(Dense(1, init='uniform', activation='sigmoid')) #cikis katmani
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=50)
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) #0.5 den buyukse 1 degilse 0 olacak.

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
