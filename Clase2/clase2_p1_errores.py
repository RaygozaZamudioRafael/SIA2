# -*- coding: utf-8 -*-

#Ante un dataset especifico es necesario realizar multiples tecnicas de 
#aprendizaje

#Regrecion lineal simple

#Para cada entrada hay una salida

#causalidad, correlacion

#Dada una coleccion de puntos obtener el fit de esa coleccion


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib auto es de python dando ventanas externas USAR EN CONSOLA

from sklearn import linear_model

#Lectura de datos

df = pd.read_csv('./countries.csv');

df_mex = df[df.country == "Mexico"]

#Seleccion de variables

x = np.asanyarray(df_mex[["year"]])
y = np.asanyarray(df_mex[["lifeExp"]]) 

#creacion del modelo
model = linear_model.LinearRegression()

#Entrenar modelo
model.fit(x,y)

#prediuccion
ypred = model.predict(x)

#ya programado
from sklearn import metrics

print("MAE:      ", metrics.mean_absolute_error(y, ypred))
print("MSE:      ", metrics.mean_squared_error(y, ypred))
print("MedAE:    ", metrics.median_absolute_error(y, ypred))
print("r2-score: ", metrics.r2_score(y, ypred))
print("EVS:      ", metrics.explained_variance_score(y, ypred))

#para no siempre llamar r2 para regrecion, en clasificacion  scoreregresa f1

print("r2-score: ", model.score(x, y))

plt.scatter(x,y)
plt.plot(x,ypred,"--r")
plt.show()

#model.predict([[2030]]) resultado array([[99x29609907]])

#Son correlacionales mas no representa la causalidad

#analisis de prediccion temporal

#Sumatoria de i=1 a n del valor absoluto de y-y^
#Da el error absoluto medio MAE [0 A Infinito]
#Error y-ypred

#Otra opcion es el MSE que es la sumatoria de 1/2n sumatpria (i=1|n)(y-y^)^2
#calculo matematico mas no aplicable

#MedianAE error absuluto mediano = media ({yi-y^i}(n|i=1)) datos atipicos
#u outlier

#-------------------------------------
#Dispersion, coheficiente de determinacion o R^(2)

#R^(2) = 1 - ((SUMATORIA(N|I=1)(Yi - Y^i)^(2))/(SUMATORIA(N|I=1)(Yi - Y^i)^(2)))

#Prediccion perfercta en R2 es 1 por lo que la metrica va de [-Inf a 1]

#0 es que el poder de explicacion del modelo es nulo

#------------------------------------------

#Coheficiente de variandza explicada

#EVS = 1-(var{Y-Y^}/var{y})


#Exxisten las verciones logaritmicas que nadie utiliza