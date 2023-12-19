#import librabries

import numpy as np   # linear algebra
import pandas as pd  # data preprocessing 
import matplotlib.pyplot as plt 
import seaborn as sns

data = pd.read_csv(r'C:\Users\SAMEER\OneDrive\Desktop\Data Science 6pm\January\26th\TASK 12 -  TASK 17\TASK-17\kc_house_data.csv')

data = data.drop(['id','date','bedrooms','bathrooms'], axis = 1)

#separating independent and dependent variable
X = data.iloc[:,1:].values
y = data.iloc[:,0].values

#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm

X=np.append(arr=np.ones((21613,1)).astype(int), values=X,axis=1)

# REGRESSION TABLE

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]

regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]

regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,16]]

regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,16]]

regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()

regressor_OLS.summary()