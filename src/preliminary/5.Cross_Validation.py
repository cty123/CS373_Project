from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd

def get_data_points():
    df = pd.read_csv("clean_data.csv", sep='\t')
    df = df.dropna(how='any', axis=0)
    x_points = df[['Distance', 'Car','BuildingArea', 'YearBuilt']]
    y_points = df[['Price']]
    return np.array(x_points), np.array(y_points)

def ridge_regression(x_train, y_train, x, y):
    # Training
    ridge = Ridge(alpha=1)
    ridge_mod = ridge.fit(x_train, y_train) 
    # Predicting
    ridge_pred = ridge_mod.predict(x)
    # Calculate standard MSE
    return np.sqrt(np.mean((ridge_pred - y) ** 2))

def lasso_regression(x_train, y_train, x, y):
    # Training
    lasso = Lasso(alpha=1)
    lasso_mod = lasso.fit(x_train, y_train) 
    # Predicting
    lasso_pred = lasso_mod.predict(x)
    # Calculate standard MSE
    return np.sqrt(np.mean((lasso_pred - y) ** 2))

# Function for performing K-fold cross validation
def k_fold_ridge(k,X,y):
    (n, d) = np.shape(X)
    z = np.zeros((k, 1))
    for i in range(0,k):
        T = set(range(int(np.floor((n*i)/k)), int(np.floor(((n*(i+1))/k)-1))+1))
        S = set(range(0, n)) - T

        z[i] = ridge_regression(X[list(S)], y[list(S)], X[list(T)], y[list(T)])
    return z

def k_fold_lasso(k,X,y):
    (n, d) = np.shape(X)
    z = np.zeros((k, 1))
    for i in range(0,k):
        T = set(range(int(np.floor((n*i)/k)), int(np.floor(((n*(i+1))/k)-1))+1))
        S = set(range(0, n)) - T

        z[i] = lasso_regression(X[list(S)], y[list(S)], X[list(T)], y[list(T)])
    return z

x, y = get_data_points()

print("================= Ridge Regression ==============")
print(k_fold_ridge(10, x, y))
print("================= Lasso Regression ==============")
print(k_fold_lasso(10, x, y))