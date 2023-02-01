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

#Caracteristicas polinomicas


#Esto decide si la linea es linear o mas polinomios
poly_features = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly_features.fit_transform(x)


#Escalar los datos
scaler = StandardScaler()
x_poly_scaled = scaler.fit_transform(x_poly)


#Modelo lineal
lin_reg = LinearRegression()
lin_reg.fit(x_poly_scaled, y)

#Prediccion forma manual
#lin_reg.predict(scaler.transform(poly_features.transform([[1.5]])))

xnew = np.linspace(-3,3,100).reshape(-1,1)
xnew_poly = poly_features.transform(xnew)
xnew_poly_scaled = scaler.transform(xnew_poly)
ynew = lin_reg.predict(xnew_poly_scaled)

print('r2: ', lin_reg.score(x_poly_scaled,y))


#Dibujo

#plotxnew
plt.plot(xnew,ynew,'-k',label='modelo')

plt.plot(x,y,".b")
plt.xlabel(r'$x_1$', fontsize=18)
plt.ylabel(r'$y$', fontsize=18)

plt.axis([-3,3,0,10])
plt.show()

#Se saca el p.f.grade de eso se procesa a S y de ahgi a LR para sacar y^ desde x