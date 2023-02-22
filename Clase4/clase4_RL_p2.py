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

data = pd.read_csv('diabetes.csv')

#x = np.asanyarray(data[['Glucose']])#Cambiar en base a las columnas
#Error de novato salida como entrada
x = np.asanyarray(data.drop(columns=['Outcome']))
y = np.asanyarray(data[['Outcome']]).ravel() #evitar warnings
#ravel hace que quede un objeto de un arreglo unidimencion mas que una matriz

#particionar los datos
xtrain, xtest, ytrain, ytest = train_test_split(x,y)

model = Pipeline([('scaler', StandardScaler()),
                  ('log_reg', LogisticRegression())]) #Solver sgd sgt?

#model = LogisticRegression()
model.fit(xtrain, ytrain)

print('Train: ', model.score(xtrain, ytrain))
print('Test:  ', model.score(xtest, ytest))

#Escalador

#Dibujar el espacio
#Como es pipeline debemnos entrar usando el nombre del modelo
coeff = list(model.named_steps['log_reg'].coef_[0])
#Se puede llamar coeff en consola para ver los resultados de las variables
labels = list(data.drop(columns='Outcome').columns)

features = pd.DataFrame()
features['Features'] = labels
#Normal, necesitamos los valores absolutos de coeff
#features['Importance'] = coeff
features['Importance'] = np.abs(coeff)
features.sort_values(by=['Importance'], inplace = True)
features.set_index('Features', inplace = True)
features.Importance.plot(kind='barh')
plt.xlabel('Importance')
plt.title('Explicacion de variables')
plt.show()



