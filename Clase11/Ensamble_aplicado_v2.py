import numpy  as np
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.ensemble         import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble         import VotingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import SVC

data = pd.read_csv('diabetes.csv')

x = np.asanyarray(data.iloc[:,:-1])
y = np.asanyarray(data.iloc[:,-1])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=0)

dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(xtrain,ytrain)

bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,
                       max_features=1.0, n_estimators=100)
bg.fit(xtrain,ytrain)

adb = AdaBoostClassifier(DecisionTreeClassifier(),
                       n_estimators=5, learning_rate=1)
adb.fit(xtrain,ytrain)

model_lr = LogisticRegression(solver = 'lbfgs', max_iter=500)
model_dt = DecisionTreeClassifier()
model_svm = SVC(kernel='rbf',gamma='scale')

evc = VotingClassifier(estimators = [('lr',model_lr),
                                     ('dt',model_dt),
                                     ('svm', model_svm)],
                       voting='hard')
evc.fit(xtrain, ytrain)

print('DT - Train:  ', dt.score(xtrain, ytrain))
print('DT - Test:   ', dt.score(xtest,  ytest))
print('--------------------------------------\n')
print('RF - Train:  ', rf.score(xtrain, ytrain))
print('RF - Test:   ', rf.score(xtest,  ytest))
print('--------------------------------------\n')
print('BG - Train:  ', bg.score(xtrain, ytrain))
print('BG - Test:   ', bg.score(xtest,  ytest))
print('--------------------------------------\n')
print('AB - Train:  ', adb.score(xtrain, ytrain))
print('AB - Test:   ', adb.score(xtest,  ytest))
print('--------------------------------------\n')
print('EVC - Train:  ', evc.score(xtrain, ytrain))
print('EVC - Test:   ', evc.score(xtest,  ytest))
print('--------------------------------------\n')










