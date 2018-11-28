from P1_Load_Data import get_data_points
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from statistics import mean
import numpy as np

def coef_determin(y_real, y_pred):
    y_bar = sum(y_real) / len(y_real)
    SE_y = sum([(y - y_bar) ** 2 for y in y_real])
    SE_line = sum([(y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real))])
    return 1. - (SE_line / SE_y)[0]

x, y = get_data_points()
ridge = Ridge(alpha=1)
ridge_mod = ridge.fit(x, y) 

ridge_pred = ridge_mod.predict(x)

print("=========== Ridge regression ============")
print("R^2: %f" % coef_determin(y, ridge_pred))
print("MSE: %f" % np.sqrt(np.mean((ridge_pred - y) ** 2)))