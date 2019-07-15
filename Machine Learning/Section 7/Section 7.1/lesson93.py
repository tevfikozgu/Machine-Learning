#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:16:45 2019

@author: tevfikozgu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] #verilerdeki n. satir 1 ise odul 1 olucak 
    toplam = toplam + odul
    

plt.hist(secilenler) #bunu anlamaya calis 
plt.show()

# acaba UCB bunu gececek mi bir sonraki derste inceliyoruz