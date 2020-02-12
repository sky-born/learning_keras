"""
1. Light GBM
https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

def Data_func():
    train_test = pd.read_csv('bd_train_test.csv')
    
    ### ['NRB','RB'] -> [0, 1]
    y = train_test['Class'].values
    encoder = preprocessing.LabelEncoder()
    y1 = encoder.fit_transform(y)
    train_test['Class_int'] = y1
    #
    
    train = train_test[train_test['Status'] == 'Train']
    test = train_test[train_test['Status'] == 'Test']

    X_train = train.drop(['CAS-RN','Smiles','Class','Status','Class_int'], axis = 1).values
    y_train = train['Class_int'].values
    
    X_test = test.drop(['CAS-RN','Smiles','Class','Status','Class_int'], axis = 1).values
    y_test = test['Class_int'].values
    
    ### standard scailing ## 2월 8일자 수정 scailing 된 값이 저장되도록 수정
    X_datas = [X_train, X_test]
    for i, datas in enumerate(X_datas):
        std_scale = preprocessing.StandardScaler().fit(datas)
        datas = std_scale.transform(datas)
        X_datas[i] = datas

    X_train, X_test = X_datas

    return X_train, y_train, X_test, y_test

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

X_train, y_train, X_test, y_test = Data_func()

###GBM hyperparameter opt (for loop)
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=42)
    gb_clf.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))    

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=1.25, max_features=2, max_depth=2, random_state=42)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))


###Precision - Recall trade off (Threshold tunning)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(gb_clf2, X_train, y_train, cv=5, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

import matplotlib.pyplot as plt

def plot_pr_re_thr(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label="precision")
    plt.plot(thresholds, recalls[:-1], 'r-', label="recall")
    plt.xlabel('threshold')
    plt.legend(loc='center left')
    plt.ylim([0,1])
    plt.show()

plot_pr_re_thr(precisions, recalls, thresholds)

y_scores_test = gb_clf2.decision_function(X_test)
y_test_pre_90 = (y_scores_test > -3.0)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pre_90))

print("Classification Report")
print(classification_report(y_test, y_test_pre_90))


###ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

plot_roc_curve(fpr,tpr)