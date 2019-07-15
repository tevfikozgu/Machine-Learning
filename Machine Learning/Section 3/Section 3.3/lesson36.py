 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 31 16:18:33 2019

@author: tevfikozgu
"""

#lesson 36 : Polinomal Regesyon (FIXED MODEL)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

veriler = pd.read_csv("maaslar.csv")

#dataframe dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPy Array Donusumu
X=x.values
Y=y.values

############################################

#Bu kisim linear Regression Yontemi
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
#Gorsellestirmegm
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.show()

#############################################

#polynomial regression 2. dereceden
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
#Gorsellestirme
plt.scatter(X,Y,color ='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)) , color = 'blue')
plt.show()

#polynomial regression 4. dereceden
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
#Gorsellestirme
plt.scatter(X,Y,color ='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)) , color = 'blue')
plt.show()

