#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:53:20 2019

@author: tevfikozgu
"""

#lesson 10 : Kategorik Veriler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("missing_values.csv")



print(veriler)


boy=veriler[["boy"]]
print(boy)

boy_kilo=veriler[["boy","kilo"]]
print(boy_kilo)

class people:
    boy=180
    def kosmak(self,b):
        return b+10
    
ali = people()
print(ali.boy)
print(ali.kosmak(90))


#sci-kit learn = sklearn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN" , strategy = "mean", axis=0)

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:1,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print("\n")
print(Yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
