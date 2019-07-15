#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:34:58 2019

@author: tevfikozgu
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv')

import re

yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][0]) # ^ not demek boylece yorumlar['Review'][6] burdaki kelimelerde [a-zA-Z] olmayanlarin hepsi ' ' oluyor.

yorum = yorum.lower() #kucuk harf yapti.
yorum = yorum.split() #kelimeleri liste haline getirdi.

import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords

yorum2 = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] #kelimeler turkceyse turkish yazabilirsin.
#29. satir gidicek yorumdaki ilk kelimeyi alicak stopword mu diye bakiak degilse kokunu alicak ve listenin elemani yapicak sirayla

yorum3 = ' '.join(yorum2) #yorum2 yi stringe donusturdu.
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i]) # ^ not demek boylece yorumlar['Review'][6] burdaki kelimelerde [a-zA-Z] olmayanlarin hepsi ' ' oluyor.
    yorum = yorum.lower() #kucuk harf yapti.
    yorum = yorum.split() #kelimeleri liste haline getirdi.
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum) #hepsini derlem isimli listede biriktirdik.
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000) #max_features en fazla kullanilan kelime sinirini belirliyo
#kelimelerin siklik araligini hesaplicak
X = cv.fit_transform(derlem).toarray() #bagimsiz degisken --- bunda satirlar yorumlar sutunler o kelimenin satirda kac tane gectigi buna spars matris deniyor
y = yorumlar.iloc[:,1].values #bagimli degisken