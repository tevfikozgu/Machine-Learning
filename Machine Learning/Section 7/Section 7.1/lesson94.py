#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:32:43 2019

@author: tevfikozgu
"""

#UCB nin farki bir oncekindeki veriye gore hareket ediyor. Yani omceki ilanda bi reklam varsa sonrakinde o on plana cikiyor.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
toplam = 0
oduller = [0] * d #Ri(n)
tiklamalar = [0] * d #o ana kadar ki tiklamalar Di(n)
secilenler = []
for n in range(1,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2* math.log(n)/tiklamalar[i]) 
            ucb = ortalama + delta
        else:
            ucb = N*10
            
        if max_ucb < ucb:
            max_ucb=ucb
            ad = i
            
    secilenler.append(ad)  
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n,ad] #verilerdeki n. satir 1 ise odul 1 olucak
    oduller[ad] = oduller[ad] + odul
    toplam += odul
    
print('Toplam Odul')
print(toplam)