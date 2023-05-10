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
#scores = cross_val_score(model,xtrain,ytrain, cv=5,scoring = 'f1_macro')
#print(scores)
#print(np.mean(scores))

#Se crea un hypermodelo En Modelos que tienen CV al final TIENEN CV
hp = {'max_depth': [1,2,3,4,5],
      'min_samples_leaf': [1,5,7,10,20],
      'min_samples_split': [2,5,8,10],
      'criterion': ['gini','entropy']
      }

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = GridSearchCV(model, hp , cv=5, scoring='f1_macro')



search.fit(xtrain,ytrain)

#Aqui el test es el test interno o el conjunto de desarrollo/validacion
#Esto entreno 1000 modelos 5 de max por 5 de min samp por 4 de min_split
#por 2 de criterion por 5 de cv en search
print(search.cv_results_['mean_test_score'])

best_model = search.best_estimator_ #esto muestra el mejor modelo obtenido

best_model.fit(xtrain, ytrain)
print('Train: ', best_model.score(xtrain, ytrain))
print('Test:  ', best_model.score(xtest, ytest))
