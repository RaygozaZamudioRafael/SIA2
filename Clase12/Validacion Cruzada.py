import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('loan_prediction.csv')

"""
Recordar siempre explorar el ds dado

df.head()
df.columns
df.shape
df.info

Tambien normalizar los datos siempre que sea posible
"""
#Normalizacion de los datos
x = np.asanyarray(df.iloc[:,:-1])
y = np.asanyarray(df.iloc[:,-1])

#Crear la particion de datos
xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=0)

#Crear modelo
model = DecisionTreeClassifier()
#                                              K
#Esto manda la 'x' y 'y' para que haga el CrossV automaticamente
scores = cross_val_score(model,xtrain,ytrain, cv=5,scoring = 'f1_macro')
#print(scores)
print(np.mean(scores))







