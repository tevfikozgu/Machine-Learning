#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.1. Veri Yukleme
veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
print(y_test)
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5 , metric = 'minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(y_pred)
cm = confusion_matrix(y_test,y_pred)
print(cm)

knn = KNeighborsClassifier(n_neighbors=5 , metric = 'minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(y_pred)
cm = confusion_matrix(y_test,y_pred)
print(cm)
#verilerin icindeki cocuk ornekler bozuyor tahmini onlari cikarip dene!!!

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC-Linear')
print(cm)

svc = SVC(kernel = 'rbf')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC-RBF')
print(cm)

svc = SVC(kernel = 'poly') #tek degisen yer rbf yerine poly yazdik
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC-RBF')
print(cm)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)



#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm) # her seferinde sonuc degisiyor dene bak!!
