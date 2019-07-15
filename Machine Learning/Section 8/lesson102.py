#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:40:05 2019

@author: tevfikozgu
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv')

import re

yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][6]) # ^ not demek boylece yorumlar['Review'][6] burdaki kelimelerde [a-zA-Z] olmayanlarin hepsi ' ' oluyor.

yorum = yorum.lower() #kucuk harf yapti.
yorum = yorum.split() #kelimeleri liste haline getirdi.