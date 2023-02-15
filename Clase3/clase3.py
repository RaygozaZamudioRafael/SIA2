"""

Metodos regularizados

    ->Sobreentrenamiento
        -Ir por mas datos
        -Un modelo mas simple
        -Regularizar


"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing   import PolynomialFeatures
from sklearn.preprocessing   import StandardScaler

from sklearn.pipeline        import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model    import Lasso
from sklearn.linear_model    import Ridge
from sklearn.linear_model    import ElasticNet

#Generar  datos.

np.random.seed(42)

m = 200;
x = 6 * np.random.rand(m,1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m,1)

#Particionar datos
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25)

xnew = np.linspace(-3,3,200).reshape(-1,1);

#Creacion de modelo L2

alpha = 0.1

"""
Datos con Ridge
datos con 0
Train:  0.8032915401498116
Test:  -1067027556.9749968

datos con 0.0000001
Train:  0.8652309677419718
Test:  0.8596454826445312

datos con 0.001
Train:  0.8480446418725651
Test:  0.8599004126698313

datos con 0.1
Train:  0.8480446418725651
Test:  0.8599004126698313

datos con 1
Train:  0.8435655753624849
Test:  0.8638809083138047

datos con 10
Train:  0.8273519055765512
Test:  0.8687455392374843

datos con 100
Train:  0.7382602927332671
Test:  0.7846211784972578

"""


model = Pipeline([('poly', PolynomialFeatures(degree=300, include_bias=False)),
                  ('scaler', StandardScaler()),
                  #('Ridge', Ridge(alpha))]) se cambio el alpha como en la foto
                  ('Lasso', Lasso(alpha))])


model.fit(xtrain,ytrain)


print('Train: ', model.score(xtrain,ytrain))
print('Test: ', model.score(xtest,ytest))

#Dibujar

ynew = model.predict(xnew)
plt.plot(xnew, ynew, 'k-', label='model', linewidth = 2)
plt.plot(xtrain,ytrain, '.b', label='train', linewidth = 3)
plt.plot(xtest, ytest, '.r', label='test', linewidth = 3)

plt.legend(loc='upper left')

plt.xlabel('x')
plt.ylabel('y')
plt.axis([-3,3,0,10])
plt.show()

