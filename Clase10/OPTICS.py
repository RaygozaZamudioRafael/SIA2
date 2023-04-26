#%matplotlib auto
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture, metrics
from sklearn.preprocessing import StandardScaler

#----------------------------Crear dataset ejemplo---------------------------

np.random.seed(0)
n_samples = 1500
X = 6* [None]

#Circulos concentricos
Xtemp, _ = datasets.make_circles(n_samples = n_samples, factor=.5, noise=0.05)
X[0]  = StandardScaler().fit_transform(Xtemp)

# Lunas
Xtemp, _ = datasets.make_moons(n_samples = n_samples, noise=0.05)
X[1]  = StandardScaler().fit_transform(Xtemp)

# Blobs
Xtemp, _ = datasets.make_blobs(n_samples = n_samples, random_state=8)
X[2]  = StandardScaler().fit_transform(Xtemp)

# Plano sin agrupaciones
Xtemp = np.random.rand(n_samples,2)
X[3]  = StandardScaler().fit_transform(Xtemp)

# Blobs con deformacion anisotróipica
Xtemp, _ = datasets.make_blobs(n_samples = n_samples, random_state=170)
Xtemp = np.dot(Xtemp,[[0.6,-0.6],[-0.4,0.8]])
X[4]  = StandardScaler().fit_transform(Xtemp)

# Blobs con diferentes varianzas
Xtemp, _ = datasets.make_blobs(n_samples = n_samples, random_state=40,
                               cluster_std=[1.0,2.5,0.5])
X[5]  = StandardScaler().fit_transform(Xtemp)

n_clusters = [2,2,3,3,3,3]

#----------------------------------------------------------------------------

##Tecnicas de agrupación
#Guardar como lista vacia
y=[]
for c, x in zip(n_clusters, X):
    model = cluster.OPTICS(min_samples=20,xi=0.05,
                           min_cluster_size=0.1)
    model.fit(x)
    if hasattr(model, 'labels_'):
        y.append(model.labels_.astype(np.int))
    else:y.append(model.predict(x))
        
##Dibujar
plt.figure(figsize= (27,9))
#bASADO EN DNESIDAD, NO NECESITA CONOCER LOS CLUSTERS A PRIORI DE LA EJECUCION
plt.suptitle('OPTICS', fontsize=42)
for i in range(6):
    ax = plt.subplot(2,3,i+1)
    ax.scatter(X[i][:,0],X[i][:,1], c = y[i])





