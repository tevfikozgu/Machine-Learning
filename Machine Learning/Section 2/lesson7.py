# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#lesson 7:data import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")

print(veriler)


boy=veriler[["boy"]]
print(boy)

boy_kilo=veriler[["boy","kilo"]]
print(boy_kilo)