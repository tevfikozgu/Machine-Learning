#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:55:12 2019

@author: tevfikozgu
"""

import pandas as pd

url = "http://www.bilkav.com/wp-content/uploads/2018/03/satislar.csv"
veriler = pd.read_csv(url)
veriler = veriler.values #arraye cevirdi
X = veriler[:,0:1]
Y = veriler[:,1]

bolme = 0.33

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=bolme)

"""
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
#print(lr.predict(X_test))

import pickle #modeli kaydetmek icin kullanilir
dosya = "model.kayit"
pickle.dump(lr,open(dosya,'wb'))

yuklenen = pickle.load(open(dosya,'rb'))
#print(yuklenen.predict(X_test)) #satir 25 ile ayni sonucu vericek

"""

import pickle #modeli kaydetmek icin kullanilir
yuklenen = pickle.load(open("model.kayit",'rb'))
print(yuklenen.predict(X_test)) #satir 25 ile ayni sonucu vericek