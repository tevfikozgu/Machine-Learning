 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 2 16:18:33 2019

@author: tevfikozgu
"""

#lesson 43 : Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

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


#verilerin olceklenmesi

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)


svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()


#print(svr_reg.predict(6.6))

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)


plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')
plt.show()

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10 , random_state = 0)
rf_reg.fit(X,Y)


plt.scatter(X,Y,color = 'red')
plt.plot(X,rf_reg.predict(X),color='blue')
plt.show()


