import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from P1_Load_Data import get_data_points

def ridge_regression():
    # Obtain training data
    x, y = get_data_points()
    x_train = x
    y_train = y

    ridge = Ridge(alpha=1)
    ridge_mod = ridge.fit(x_train, y_train) 
    ridge_pred = ridge_mod.predict(x)

    print("======================= Ridge Regression =================")
    print("Standard MSE(mean square error): %f" % np.sqrt(np.mean((ridge_pred - y) ** 2)))
    print("MSE: %f" % np.mean((ridge_pred - y) ** 2))

def lasso_regression():
    # Obtain training data
    x, y = get_data_points()
    x_train = x
    y_train = y

    lasso = Lasso(alpha=1)
    lasso_mod = lasso.fit(x_train, y_train) 
    lasso_pred = lasso_mod.predict(x)

    print("======================= Lasso Regression =================")
    print("Standard MSE(mean square error): %f" % np.sqrt(np.mean((lasso_pred - y) ** 2)))
    print("MSE: %f" % np.mean((lasso_pred - y) ** 2))

ridge_regression()
lasso_regression()
