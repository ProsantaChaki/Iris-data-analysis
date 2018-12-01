#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:41:27 2018

@author: chaki
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris_Data.csv', header = None )
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regrassor_Linear = LinearRegression()
regrassor_Linear.fit(X_train, Y_train)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
regrassor_Logistic = LogisticRegression()
regrassor_Logistic.fit(X_train, Y_train)

#fitting Naive Byaes classification to the training set
from sklearn.naive_bayes import GaussianNB
classifier_naive = GaussianNB()
classifier_naive.fit(X_train, Y_train)

# fitting Support Vector Machine's to the training set
from sklearn.svm import SVC
classifier_svc = SVC()
classifier_svc.fit(X_train, Y_train)

#fitting K-Nearest Neighbours to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier_K = KNeighborsClassifier(n_neighbors=8)
classifier_K.fit(X_train, Y_train)

# Fitting Decision Tree's to the training set
from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier()
classifier_tree.fit(X_train, Y_train)


# Predicting the Test set results
Linear_pred_ = regrassor_Linear.predict(X_test)
Linear_pred = []
for i in Linear_pred_:
    Linear_pred.append( abs(round(i)))



#Accuracy score calculation
from sklearn.metrics import accuracy_score
print('accuracy of Logiatic regression is',accuracy_score(regrassor_Logistic.predict(X_test), Y_test))
print('accuracy of Linear regression is',accuracy_score(Linear_pred, Y_test))
print('accuracy of Naive Byaes classifier is',accuracy_score(classifier_naive.predict(X_test), Y_test))
print('accuracy of Support Vector machine is',accuracy_score(classifier_svc.predict(X_test), Y_test))
print('accuracy of K-Nearest Neighbours is',accuracy_score(classifier_K.predict(X_test), Y_test))
print('accuracy of Decision Tree is',accuracy_score(classifier_tree.predict(X_test), Y_test))

'''  
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Logistic_pread)'''
