#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:23:01 2019

@author: tevfikozgu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

veriler = pd.read_csv("odev_tenis.csv")

information1 = veriler.iloc[:,1:2].values
information2 = veriler.iloc[:,2:3].values

#Nominal - Ordinal ---> Numeric yapiyor

hava = veriler.iloc[:,0:1].values
#print(ulke)

le = LabelEncoder()
hava[:,0] = le.fit_transform(hava[:,0])
#print(ulke)
ohe = OneHotEncoder(categorical_features="all")
hava=ohe.fit_transform(hava).toarray()
#print(ulke)


play = veriler.iloc[:,-1:].values
le = LabelEncoder()
play = le.fit_transform(play[:,0])

windy = veriler.iloc[:,3:4].values
le = LabelEncoder()
windy = le.fit_transform(windy[:,0])


hava_bilgisi = pd.DataFrame ( data = hava, index = range(14) , columns = ["Overcast","Rainy","Sunny"])
play_bilgisi = pd.DataFrame( data=play , index = range(14) , columns = ["Play"])
windy_bilgisi = pd.DataFrame(data = windy , index=range(14) , columns=['Windy'])
sicaklik_bilgisi = pd.DataFrame(data = information1 , index=range(14) , columns=['Temperature'])
nem_bilgisi = pd.DataFrame(data = information2 , index=range(14) , columns=['Numidity'])

'''
veriler2 = veriler.apply(LabelEncoder().fit_transform) 
#bu ustteki kod tum kodlari label encoder haline donusturuyor. Yalniz Hepsini yapiyor!!! !!!!
'''
 
#DataFrame Birlestirme Islemi
s=pd.concat([hava_bilgisi,sicaklik_bilgisi],axis=1)
s1=pd.concat([s,windy_bilgisi],axis=1)
s2 = pd.concat([s1,play_bilgisi],axis=1)


#Verilerin Train ve Test Olarak Bolunmesi
x_train, x_test, y_train , y_test = train_test_split(s2,nem_bilgisi,test_size=0.33, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
#print(y_pred)



X = np.append(arr = np.ones((14,1)).astype(int), values=s2, axis=1 )
X_l = s2.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = nem_bilgisi, exog = X_l)
r = r_ols.fit()
print(r.summary())


#Calistirinca  x5'in p-value si 0.05 ten buyuk ciktigi icin siliyoruz. Buna Backward Elimination yapmak!!!
'''
X_l = s2.iloc[:,[3]].values
r_ols = sm.OLS(endog = nem_bilgisi, exog = X_l)
r = r_ols.fit()
print(r.summary())
'''
'''
new = pd.DataFrame (data=X_l,index = range(14) , columns = ["Temperature"])
x_train, x_test, y_train , y_test = train_test_split(new,nem_bilgisi,test_size=0.33, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
#print(y_pred)
'''
'''
SOR!!!
'''
