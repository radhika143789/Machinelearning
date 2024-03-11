# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset (replace with your actual data file)
data = pd.read_csv('/PNB.NS_data.csv')

# Feature selection (assuming 'Closing Price' is the target variable)
X = data[['Open', 'High', 'Close']]
y = data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=0.1)
knn_reg = KNeighborsRegressor(n_neighbors=5)

# Fit models to training data
linear_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)
lasso_reg.fit(X_train, y_train)
knn_reg.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)
y_pred_knn = knn_reg.predict(X_test)

# Evaluate model performance
print(f"Linear Regression MAE: {mean_absolute_error(y_test, y_pred_linear)}")
print(f"Ridge Regression MAE: {mean_absolute_error(y_test, y_pred_ridge)}")
print(f"Lasso Regression MAE: {mean_absolute_error(y_test, y_pred_lasso)}")
print(f"KNN MAE: {mean_absolute_error(y_test, y_pred_knn)}")
