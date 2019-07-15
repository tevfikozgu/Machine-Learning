import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) # kmeans in basari degeri 
    
plt.plot(range(1,11),sonuclar) # cizime bak 4 te egim cok az degismeye basliyor orasi k degeri olur yani 4

#WCSS bir formul bu formulle en iyi k degerinini buluyoruz.