# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 07:19:00 2023

@author: usuario

Regresion logistica y soffmax

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

iris = load_iris()

#lo de entre '' se conotro dentro del dataset llamando iris en consola y del
#texto que sale de ahi

x = iris['data']
y = iris['target']

#Se puede checar con x.shape y.shape

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

model = Pipeline([('scaler', StandardScaler()),
                  ('Softmax', LogisticRegression(multi_class = 'multinomial'))])

model.fit(xtrain, ytrain)

print('Train: ', model.score(xtrain,ytrain))
print('Test:  ', model.score(xtest,ytest))

from sklearn.metrics import confusion_matrix, classification_report

ypred = model.predict(xtest)

print('Confusion matrix: \n', confusion_matrix(ytest, ypred))
print('Classification report: \n', classification_report(ytest, ypred))