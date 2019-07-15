#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#2.1. Veri Yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("satislar.csv")
#print(veriler)

aylar = veriler[['Aylar']]
#print(aylar)
satislar = veriler[['Satislar']]
#print(satislar)

'''
BU DA DIGER SECENEK
satislar2 = veriler.iloc[:,:1].values
#print(satislar2)
'''


#verilerin egitim ve test icin bolunmesi

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)


#verilerin olceklenmesi

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


##########################

#model insaasi (Linear Regression)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)


    
    

