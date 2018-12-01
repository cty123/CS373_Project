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
    
    y_flat = y.flatten()
    sq_err = [ (y_flat[i] - ridge_pred[i]) ** 2 for i in range(len(y_flat))]
    
    print("======================= Ridge Regression =================")
    print("Standard MSE(mean square error): %f" % np.sqrt(np.mean(sq_err)))
    print("MSE: %f" % np.mean(sq_err))

def lasso_regression():
    # Obtain training data
    x, y = get_data_points()
    x_train = x
    y_train = y

    lasso = Lasso(alpha=1)
    lasso_mod = lasso.fit(x_train, y_train) 
    lasso_pred = lasso_mod.predict(x)
    
    y_flat = y.flatten()
    sq_err = [ (y_flat[i] - lasso_pred[i]) ** 2 for i in range(len(y_flat))]

    print("======================= Lasso Regression =================")
    print("Standard MSE(mean square error): %f" % np.sqrt(np.mean(sq_err)))
    print("MSE: %f" % np.mean(sq_err))

ridge_regression()
lasso_regression()
