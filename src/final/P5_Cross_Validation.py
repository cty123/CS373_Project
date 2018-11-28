from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from P1_Load_Data import get_data_points
import numpy as np

# Training with ridge regression
def ridge_regression(x_train, y_train, x, y):
    # Training
    ridge = Ridge(alpha=1)
    ridge_mod = ridge.fit(x_train, y_train) 
    # Predicting
    ridge_pred = ridge_mod.predict(x)
    # Calculate standard MSE
    return np.sqrt(np.mean((ridge_pred - y) ** 2))

# Training with lasso regression
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

if __name__ == "__main__":
    # Get data points for cross validation
    x, y = get_data_points()

    # Print out the result of cross validation
    print("================= Ridge Regression Standard MSE ==============")
    print(k_fold_ridge(10, x, y))
    print("================= Lasso Regression Standard MSE ==============")
    print(k_fold_lasso(10, x, y))