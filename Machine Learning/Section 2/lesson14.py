#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:18:33 2019

@author: tevfikozgu
"""

#lesson 14 : Veri On isleme Sablonu (FIXED MODEL)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

veriler = pd.read_csv("missing_values.csv")

"""

Object Oriented Programming

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

"""
#sci-kit learn = sklearn



#Eksik Veriler

imputer = Imputer(missing_values="NaN" , strategy = "mean", axis=0)

Yas = veriler.iloc[:,1:4].values
#print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)



#Nominal - Ordinal ---> Numeric yapiyor

ulke = veriler.iloc[:,0:1].values
#print(ulke)

le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
#print(ulke)

ohe = OneHotEncoder(categorical_features="all")
ulke=ohe.fit_transform(ulke).toarray()
#print(ulke)



#Pandas kullanicaz
#NumPy Dizilerinin DataFrame Donusumu

sonuc = pd.DataFrame ( data = ulke, index = range(22) , columns = ["fr","tr","us"])
#print(sonuc)

sonuc2 = pd.DataFrame( data=Yas , index = range(22) , columns = ["boy" , "kilo" , "yas"])
#print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
#print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index=range(22) , columns=['cinsiyet'])
#print(sonuc3)

"""
HATAYI GOR!!!!
s=pd.concat([sonuc,sonuc2])
print(s)
"""



#DataFrame Birlestirme Islemi
s=pd.concat([sonuc,sonuc2],axis=1)
#print(s)

s2 = pd.concat([s,sonuc3],axis=1)
#print(s2)



#Verilerin Train ve Test Olarak Bolunmesi
x_train, x_test, y_train , y_test = train_test_split(s,sonuc,test_size=0.33, random_state=0)

'''
x_train ile y_train sirasiyla s ve sonucun 2/3 luk kismi oldu. 
x_test ile t_test sirasiyla s ve sonucun 1/3 luk kismi oldu.
2/3 ile 1/3 olmasini test_size sagladi.
random_state de random almasini sagladi.
'''



#Verilerin Olceklenmesi. Normalization ve Standartization var. Ama biz Standartization kullaniyoruz!!!
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
#print(X_test)

'''
Standart Scaler:
sayidan ortalama degeri cikariyor ve standart sapmaya boluyor.
Bunun amaci boy kilo yasi ortak bi evren olarak dusunme. 180 ile 30 u ayni evrende karsilastirmak mantiksiz.
'''
