# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:38:55 2019

@author: Shashank
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import quandl

# Importing the Dataset
data = quandl.get('WIKI/FB')
dataset = data[['Adj. Close']]
dataset.head()
#data = pd.read_excel('data_akbilgic.xlsx')

# Defining a Variable to predict 'n' days out in future
n = 30

# Creating a Column 'target' for dependent variable shifted 'n' units up
dataset['Prediction'] = dataset[['Adj. Close']].shift(-n)
dataset.head().append(dataset.tail())

# Creating the Independent and Dependent Datasets
x = np.array(dataset.drop(['Prediction'], 1)) # Also converting dataset into numpy array
x = x[: - n] # removing the last n rows
print(x)
y = np.array(dataset['Prediction'])
y = y[: - n] # removing the last n rows
y

# Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting the Linear Regression model to the Training Set
from sklearn.linear_model import LinearRegression
reg_lr = LinearRegression()
reg_lr.fit(x_train, y_train)

# Testing the Linear Regression Model (Confidence)
lr_confidence = reg_lr.score(x_test, y_test)
lr_confidence

# Fitting the SVM Model to the Training Set
from sklearn.svm import SVR
reg_svr = SVR(kernel = 'rbf', C = 1000, gamma = 0.1)
reg_svr.fit(x_train, y_train)

# Testing the SVM Model (Confidence)
svm_confidence = reg_svr.score(x_test, y_test)
svm_confidence

# Setting a Variable equal to the Last 30 rows from the dataset['Adj. Close']
y_pred = np.array(dataset.drop(['Prediction'], 1))
y_pred = y_pred[-n :]
y_pred

# Predicting the Next 'n' days for the Linear Regression Model
lr_prediction = reg_lr.predict(y_pred)
print(lr_prediction)

# Predicting the Next 'n' days for the SVM Model
svr_prediction = reg_svr.predict(y_pred)
print(svr_prediction)
