# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib auto es de python dando ventanas externas USAR EN CONSOLA

#Regresion polinomial (regrecion lineal+truco = regrecion no lineal)
#y^ = B0 + B1 se convierte a B0 +B1X + B2X^(2)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Generar datos sinteticos

np.random.seed(42)

m = 100;
x = 6 * np.random.rand(m,1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m,1)

#Particionar datos
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25)

#modelo
degree = 2

model = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                  ('scaler', StandardScaler()),
                  ('lin-reg', LinearRegression())])

model.fit(xtrain,ytrain)

print('Train: ', model.score(xtrain,ytrain))
print('Test: ', model.score(xtest,ytest))

xnew = np.linspace(-3,3,100).reshape(-1,1)
ynew = model.predict(xnew)

#Dibujo

#plotxnew
plt.plot(xnew,ynew,'-k',label='modelo')

plt.plot(xtrain,ytrain,".b")
plt.plot(xtest,ytest,".r")
plt.xlabel(r'$x_1$', fontsize=18)
plt.ylabel(r'$y$', fontsize=18)

plt.axis([-3,3,0,10])
plt.show()

#Se saca el p.f.grade de eso se procesa a S y de ahgi a LR para sacar y^ desde x