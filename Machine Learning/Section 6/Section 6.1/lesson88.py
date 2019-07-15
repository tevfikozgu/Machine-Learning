#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:40:55 2019

@author: tevfikozgu
"""

#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv',header=None)

t= []
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)

#print(kurallar) object oldugu icin basilmaz

print(list(kurallar)) #bastirdigina bak herb&pepper alanlar ground beef de aliyor.

#eclat apriorinin bir versiyonu ama buyuk verilerde duzgun calismiyor apriori daha iyi