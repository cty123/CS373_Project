from P1_Load_Data import get_data_points
from sklearn.linear_model import Ridge, Lasso
from statistics import mean
import numpy as np

def coef_determin(y_real, y_pred):
    y_bar = sum(y_real) / len(y_real)
    SE_y = sum([(y - y_bar) ** 2 for y in y_real])
    SE_line = sum([(y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real))])
    return 1. - (SE_line / SE_y)[0]

x, y = get_data_points()
ridge = Ridge(alpha=0)
ridge_mod = ridge.fit(x, y) 

ridge_pred = ridge_mod.predict(x)
y_flat = y.flatten()
sq_err = [ (y_flat[i] - ridge_pred[i]) ** 2 for i in range(len(y_flat))]

print("Ridge regression")
print("R^2: %f" % coef_determin(y, ridge_pred))
print("Standard MSE: %f" % np.sqrt(np.mean(sq_err)))

lasso = Lasso(alpha=1e3)
lasso_mod = lasso.fit(x, y) 

lasso_pred = lasso_mod.predict(x)
y_flat = y.flatten()
sq_err = [ (y_flat[i] - lasso_pred[i]) ** 2 for i in range(len(y_flat))]

print("Lasso regression")
print("R^2: %f" % coef_determin(y, lasso_pred))
print("Standard MSE: %f" % np.sqrt(np.mean(sq_err)))