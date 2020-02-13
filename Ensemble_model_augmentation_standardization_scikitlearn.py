import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('bio_1_train.csv')
test = pd.read_csv('bio_1_test.csv')
X_train=train.iloc[:,4:]
y_train=train.iloc[:,2]
X_test=test.iloc[:,4:]
y_test=test.iloc[:,2]
X_train

X_test


y_train


y_test


from sklearn.preprocessing import MinMaxScaler


min_max_scaler = MinMaxScaler()
fitted_train = min_max_scaler.fit(X_train)
fitted_test = min_max_scaler.fit(X_test)
fitted_train



X_train_m=min_max_scaler.transform(X_train)
X_test_m=min_max_scaler.transform(X_test)
X_train_m


lr_model=LogisticRegression(solver='lbfgs',multi_class='multinomial')
knn_model=KNeighborsClassifier(n_neighbors=3)
svc_model=LinearSVC()
dt_model=DecisionTreeClassifier()


ensemble=VotingClassifier(
            estimators=[('lr',lr_model),
                        ('knn',knn_model),
                        ('svc',svc_model),
                        ('dt',dt_model)])

lr_model.fit(X_train_m, y_train)
knn_model.fit(X_train_m, y_train)
svc_model.fit(X_train_m, y_train)
dt_model.fit(X_train_m,y_train)
ensemble.fit(X_train_m, y_train)


print('LR:',lr_model.score(X_train_m, y_train))
print('KNN:',knn_model.score(X_train_m, y_train))
print('SVC:',svc_model.score(X_train_m, y_train))
print('DT:',dt_model.score(X_train_m, y_train))
print('Ensemble:',ensemble.score(X_train_m, y_train))



print('LR:',lr_model.score(X_test_m, y_test))
print('KNN:',knn_model.score(X_test_m, y_test))
print('SVC:',svc_model.score(X_test_m, y_test))
print('DT:',dt_model.score(X_test_m, y_test))
print('Ensemble:',ensemble.score(X_test_m, y_test))



