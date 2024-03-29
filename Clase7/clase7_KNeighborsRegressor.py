#KNN Actualizacion del modelo de la clase 5
#%matplotlib auto

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#iMPORT DEL MODELO
from sklearn.neighbors import KNeighborsRegressor


#Generar datos
np.random.seed(42)
m = 200
r = 0.5
ruido = r * np.random.rand(m,1)
x = 6 * np.random.rand(m,1)-3
y = 0.5 * x**2 + x +2 + ruido

#particionar datos
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

#Entrenar modelo
model = KNeighborsRegressor(n_neighbors=25, weights='uniform', metric='minkowski')
#Menos vecinos, sobre entrenada, mas vecinos, subEntrenada
#Dada la naturaleza del extrapolador, no puede encontrar los vecinos suficiente
#en los extremos ocacionando que esta se estanque

model.fit(xtrain,ytrain)
train_score = model.score(xtrain,ytrain)
test_score = model.score(xtest,ytest)

#dibujo
plt.figure()
xnew = np.linspace(-3,3,200).reshape(-1,1)
ynew = model.predict(xnew)
plt.plot(xtrain, ytrain, 'b.', label='Train')
plt.plot(xtest, ytest, 'r.', label='Test')
plt.plot(xnew, ynew,'k-',linewidth=3,label='Model')
plt.xlabel(r'$x_1$', fontsize=18) # $escriubir texto matematico en codigo de latex
plt.ylabel(r'$y$', fontsize=18)
plt.title('MLP Regressor')
plt.legend(loc='upper left')
plt.text(2.5,9, '%.2f' % train_score,size=15, horizontalalignment='right')
plt.text(2.5,8.5, '%.2f' % test_score,size=15, horizontalalignment='right')
plt.show()

