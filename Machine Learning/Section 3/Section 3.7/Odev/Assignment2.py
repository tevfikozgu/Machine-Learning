import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm

veriler = pd.read_csv("maaslar_yeni.csv")

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

print(veriler.corr())


lin_reg = LinearRegression()
lin_reg.fit(X,Y)
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())


#Polynomial Regression 2. dereceden
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

#verilerin olceklenmesi
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)


#SVR Regression
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
print("R-Square SVR Regression Degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))
#print(svr_reg.predict(6.6))

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())
print("R-Square Decision Tree Regression Degeri:")
print(r2_score(Y, r_dt.predict(X)))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10 , random_state = 0)
rf_reg.fit(X,Y)
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
print("R-Square Random Forest Regression Degeri:")
print(r2_score(Y, rf_reg.predict(X)))