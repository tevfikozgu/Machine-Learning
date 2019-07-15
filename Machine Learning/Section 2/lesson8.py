#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:11:40 2019

@author: tevfikozgu
"""

#week 8:Object Oriented Programming

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")

print(veriler)


boy=veriler[["boy"]]
print(boy)

boy_kilo=veriler[["boy","kilo"]]
print(boy_kilo)

class people:
    boy=180
    def kosmak(self,b):
        return b+10
    
ali = people()
print(ali.boy)
print(ali.kosmak(90))

l = [1,2,3,4] #list