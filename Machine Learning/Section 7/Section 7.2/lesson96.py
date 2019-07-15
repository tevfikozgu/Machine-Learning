#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:48:05 2019

@author: tevfikozgu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
toplam = 0 #toplam odul
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for n in range(1,N):
    ad = 0 #secilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate(birler[i]+1, sifirlar[i]+1)
        if rasbeta > max_th:
            max_th=rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n,ad] #verilerdeki n. satir 1 ise odul 1 olucak
    if odul == 1:
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad]+1
    toplam += odul
    
print('Toplam Odul')
print(toplam)