#Se vera la libreria pandas

import numpy as np
import pandas as pd

#data = {'Nombre':['carlos','Julia','Fabiola','nombre'],
#        'edad': [28,25,56,11],
#        'calificacion': [100,89,48,35]
#        }

df = pd.read_csv('countries.csv');

#se puede usar .head() y .tail() para checar los primeros y ultimos datos
#df.shape dara la frma de la tabla y con .colums dara el nombre y tipo de las cabeceras

#.info() dara la informacion de todo lo del dataframe, permite checar si todos los registros
#estan enteros y de esta manera actuar para rellenar los datos especificados

#Las cadenas y tipos iterativos son declarados como tipo objeto, checar que los
#numericos sean numericos

#.describe() dara el registro de las variables numericas

#df.values regresara un arreglo de tipo numpy, si tiene texto lo regreasara 
#como objeto

#Todas las funciones de excel numericas, tipo mean() max() min()

#df = df.rename(columns={'gdpPercap':'gdp'})

#sacar una sola columna usar df['nombreCabecera'] para sacar la serie


#df.country puede causar errores en caso de que exista metodo o atributo

#df[['country´]] recupera el dataframe como dataframe y no como serie

#df2 = df.replace(1952, 'one') crea un nuevo dataframe y editarlo

#Sacar la mascara con vectores bool para saber que filtrar
#df_mex = df[df.country == 'Mexico']

#Filtros multiples

#Forma correcta
#df[(df.country == 'Mexico')&(df.year >1977)&(df.lifeExp > 70)]

#Forma rapida
#df[df.country == 'Mexico'][df.year >1977][df.lifeExp > 70]
#Se enmascaran usando mascaras sobre mascaras, el problema es que son mascaras
#grandes del df cuando se aplica a la segunda mascara estamos en un df mas
#pequeño que el df donde la segunda mascara se saco

#Resetear los indices del ds

#df_mex = df_mex.reset_index()
#crea una columna llamada index donde mantiene los viejos indices
#df_mex.drop('index',axis1,inplace=True)

#ordenar.sort_values('cabecera')

#graficos
#df.hist() para el histograma
#%matplotlib auto es de python dando ventanas externas

#df.plot(x='anio',y='gdp')

#pd.plotting.scatter_matrix(df) función de pandas que recibe un dataframe

#list(df[(df.year == 2002)&(df.lifeExp >= 80)].country)
#len(list(df[(df.year == 2002)&(df.lifeExp >= 80)].country))

#df[df.country == 'Kuwait'].plot(x='year',y='gdp')

#df[(df.gdp == df.gdp.max())]


#df[(df.country == 'Mexico')&(df['pop'] > 70000000)].sort_values('pop').year.iat[0]

