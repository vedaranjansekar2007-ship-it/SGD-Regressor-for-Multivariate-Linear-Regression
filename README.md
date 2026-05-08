# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California housing dataset and select input and output variables.
2. Split the dataset into training and testing data, then standardize the values.
3. Train the Multi-Output SGD Regression model using the training data.
4. Predict the output for test data and display the predicted values.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VEDARANJAN S
RegisterNumber:  212225220119
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
x=data.data[:, :3]
y=np.column_stack((data.target,data.data[:,6]))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.fit_transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.fit_transform(y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-6)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
print("\nPredictions:\n",y_pred[:5])


```

## Output:

<img width="1032" height="252" alt="image" src="https://github.com/user-attachments/assets/bfcaad8b-139a-4489-b235-ddcea1befbc3" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
