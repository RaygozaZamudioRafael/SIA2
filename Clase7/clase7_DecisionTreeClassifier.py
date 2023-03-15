import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split

#Import model
from sklearn.tree import DecisionTreeClassifier

#Generate datasets
rng = np.random.RandomState(2)
ruido = 1
x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1)
x += ruido * rng.uniform(size=x.shape)

#Crear el dataset
datasets = [make_moons(noise=0.1),
                     make_circles(noise=0.1, factor=0.5),
                     (x,y)]

#iterate over dataset
figure = plt.figure(figsize=(9,3))
h=0.02 #Entre mas chica mas costo, aumento cuadratiuco
cm = plt.cm.RdBu

cm_bright = ListedColormap(['#FF0000','#0000FF'])

for ds_cont, ds in enumerate(datasets):
    
    x, y = ds
    x = StandardScaler().fit_transform(x) 
    #el fit_transform es el concatenamiento para que standardscaler regrese algo
    xtrain,xtest,ytrain,ytest = train_test_split(x,y)
    
    #graph limits la x no es la x que entrenamos
    #Genera el margen durante el dibujado
    xmin,xmax = x[:,0].min() - 0.05, x[:,0].max() + 0.05
    ymin,ymax = x[:,1].min() - 0.05, x[:,1].max() + 0.05
    
    xx,yy = np.meshgrid(np.arange(xmin,xmax,h),
                        np.arange(ymin,ymax,h))

    #Create model and fit
    model = DecisionTreeClassifier(min_samples_leaf=20)
    model.fit(xtrain,ytrain)
    
    #Calculate scores
    score_train = model.score(xtrain, ytrain)
    score_test = model.score(xtest, ytest)
    
    # Draw model
    
    ax = plt.subplot(1,3,ds_cont+1) #regilla (1,3) y el contabilizador
    
    if hasattr(model, 'decision_function'):
        zz = model.decision_function(np.c_[xx.ravel(),yy.ravel()]) #concatenate es reducido a c_
    else:
        zz = model.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
    
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap = cm, alpha = 0.8) #alpha es la transparencia, siempre se llama asi en estos modelos
    
    ax.scatter(xtrain[:,0],xtrain[:,1], c=ytrain, cmap = cm_bright,
               edgecolor='k') #puntos de borde
    
    ax.scatter(xtest[:,0],xtest[:,1], c=ytest, cmap = cm_bright,
               edgecolor='k', alpha = 0.6) #puntos de prueba
    
    ax.text(xmax-0.3, ymin+0.7, '%.2f' % score_train, size=15,
            horizontalalignment = 'right') #harcodeado es donde ira el texto
    ax.text(xmax-0.3, ymin+0.3, '%.2f' % score_train, size=15,
            horizontalalignment = 'right')
    
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_yticks(())
    ax.set_yticks(())

plt.tight_layout()
plt.show()

# usar %matplotlib auto