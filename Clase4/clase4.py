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

#data da los datos | data.info() da las columnas

#%matplotlib auto
#pd.plotting.scatter_matrix(data)
#Las nubes dispersas de barras demuestran una recolecta de datos pobre y dara 
#un score bajo, esto es el sanitycheck

corr = data.corr()

import seaborn as sns #mas bonita que mathplotlib basada en la misma
#otra opcion es ggplot para hacer cosas bonitas
sns.heatmap(corr,
            xticklabels = data.columns,
            yticklabels = data.columns)

#Mas brillante fuera de diagonal muestra la correlacion

#Todo lo anterior es ciencia de datos y nos permite detectar errores en los 
#datos antes de trabajarlos

#Verificar el Dtypecon data.info(), si hay texto es que puede que este
#corrupto en una col numerica se inserto un null