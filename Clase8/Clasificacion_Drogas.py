import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data = pd.read_csv('drug200.csv')

#Agrupar columna a clasificar
print('--------------------\n\n')
#Cambio de la columna edad por la nueva columna que los junta en una categoria
data['Na_to_K_Bigger_Than_15'] = [1 if i >=15.015 else 0 for i in data.Na_to_K]
print('Data info ')
print(data.info())

#Cambiar datos a int
def label_encoder(y):
    le = LabelEncoder()
    data[y] = le.fit_transform(data[y])

label_list = ["Sex","BP","Cholesterol","Na_to_K","Na_to_K_Bigger_Than_15","Drug"]

for l in label_list:
    label_encoder(l)

print('******************\n\n')
print(data.head())

#Separar datos y preparar set de entrenamiento
print('--------------------\n\n')
x = data.drop(["Drug"],axis=1)
y = data.Drug

xtrain, xtest, ytrain, ytest = train_test_split(x,y,
                                                    test_size = 0.2, 
                                                    random_state = 42, 
                                                    shuffle = True)

ytrain = ytrain.values.reshape(-1,1).ravel()
ytest = ytest.values.reshape(-1,1).ravel()

print("x_train shape:",xtrain.shape)
print("x_test shape:",xtest.shape)
print("y_train shape:",ytrain.shape)
print("y_test shape:",ytest.shape)

print('--------------------\n\nARBOL')

model = DecisionTreeClassifier(min_samples_leaf=11)
model.fit(xtrain,ytrain)
    
#Calculate scores
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)

print('--------------------\n\n')

print('Entrenamiento: ' , score_train)
print('Prueba: ' , score_test)

print('--------------------\n\nKNeightbors')

model = KNeighborsClassifier(n_neighbors=50)
model.fit(xtrain,ytrain)

#Calculate scores
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)

print('--------------------\n\n')

print('Entrenamiento: ' , score_train)
print('Prueba: ' , score_test)


print('--------------------\n\nSVC')

model = SVC(kernel='rbf',C=1, gamma = 10)
model.fit(xtrain,ytrain)
 
#Calculate scores
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)

print('--------------------\n\n')

print('Entrenamiento: ' , score_train)
print('Prueba: ' , score_test)










