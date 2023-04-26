# -*- coding: utf-8 -*-
"""
Agrupación / Clustering





"""
#%matplotlib auto
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

#Por cuestiones de bibloteca solo abre jpg, png no lo reconoce
I = mpimg.imread('Fondopantalla.jpg')
I = np.array(I, dtype=np.float64)/ 255.0

w,h,d = I.shape
n_clusters = 3

I_array = np.reshape(I, (w*h,d))

#Tomar muestra de la imagen para reducir la cantidad de pixeles
I_sample = shuffle(I_array)[:2000]

#Crear modelo y entrenarlo con fit
model = KMeans(n_clusters = n_clusters).fit(I_sample)
labels = model.predict(I_array)

#Nueva imagen
I_labels = np.reshape(labels, (w,h))
#Imagen salida
I_out = np.zeros((w,h,d))
label_idx = 0
for i in range(w):
    for j in range(h):
        I_out[i,j,:] = model.cluster_centers_[I_labels[i,j]]

plt.figure()
plt.subplot(1,2,1)
plt.title('Imagen original')
plt.imshow(I)
plt.subplot(1,2,2)
plt.title('Imagen cuantizada')
plt.imshow(I_out)