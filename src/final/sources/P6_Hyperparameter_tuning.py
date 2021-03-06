from scipy.stats import mode
from P1_Load_Data import get_data_points
from sklearn.linear_model import Ridge, Lasso
import numpy as np

def bootstrapping_ridge(B, X_subset, y_subset, alpha):
     # Get the size of the data
    n = len(X_subset)
    # Have an array to record errors
    bs_err = np.zeros(B)
    # Repeat B times
    for b in range(B):
        # Randomly select n samples
        train_samples = list(np.random.randint(0,n,n))
        # Construct test samples
        test_samples = list(set(range(n)) - set(train_samples))
        # Construct learning model
        ridge = Ridge(alpha=alpha)
        # Start learning
        ridge.fit(X_subset[train_samples], y_subset[train_samples])
        # Export result of test data
        ridge_pred = ridge.predict(X_subset[test_samples])
        # Store error in best error array 
        y_flat = y_subset[test_samples].flatten()
        sq_err = [ (y_flat[i] - ridge_pred[i]) ** 2 for i in range(len(y_flat))]
        bs_err[b] = np.sqrt(np.mean(sq_err))
    
    err = np.mean(bs_err)
    return err

def hyper_tuning_ridge():
    # Get data points
    x, y = get_data_points()
    # Get data shape
    (n, d) = np.shape(x)
    # Set k = 10 because we are still doing 10-fold
    k = 10
    # Best alpha_list 
    best_alphas = []

    # Do k-folds
    for i in range(k):
        # Alpha candidates
        alpha_arr = [-1e2, -1e1, 0, 1e1, 1e2, 1e3, 1e4, 1e5]
        # Store best alpha
        best_a = None
        # Store best err
        best_err = None

        T = set(range(int(np.floor((n*i)/k)), int(np.floor(((n*(i+1))/k)-1))+1))
        S = set(range(0, n)) - T
        
        for a in alpha_arr:
        
            e = bootstrapping_ridge(30, x[list(S)], y[list(S)], a)
            
            if best_err is None or e < best_err:
                best_err = e
                best_a = a
                
        print("Fold #%d: Best error is %f" % (i, best_err))
        best_alphas.append(best_a)
        
    print(best_alphas)
    print("The best alpha is %f" % mode(best_alphas).mode)

def bootstrapping_lasso(B, X_subset, y_subset, alpha):
   # Get the size of the data
    n = len(X_subset)
    # Have an array to record errors
    bs_err = np.zeros(B)
    # Repeat B times
    for b in range(B):
        # Randomly select n samples
        train_samples = list(np.random.randint(0,n,n))
        # Construct test samples
        test_samples = list(set(range(n)) - set(train_samples))
        # Construct learning model
        lasso = Lasso(alpha=alpha)
        # Start learning
        lasso.fit(X_subset[train_samples], y_subset[train_samples])
        # Export result of test data
        lasso_pred = lasso.predict(X_subset[test_samples])
        # Store Standard MSE
        y_flat = y_subset[test_samples].flatten()
        sq_err = [ (y_flat[i] - lasso_pred[i]) ** 2 for i in range(len(y_flat))]
        bs_err[b] = np.sqrt(np.mean(sq_err))
    
    err = np.mean(bs_err)
    return err

def hyper_tuning_lasso():
    # Get data points
    x, y = get_data_points()
    # Get data shape
    (n, d) = np.shape(x)
    # Set k = 10 because we are still doing 10-fold
    k = 10
    # Best alpha_list 
    best_alphas = []

    # Do k-folds
    for i in range(k):
        # Alpha candidates
        alpha_arr = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
        # Store best alpha
        best_a = None
        # Store best err
        best_err = None

        T = set(range(int(np.floor((n*i)/k)), int(np.floor(((n*(i+1))/k)-1))+1))
        S = set(range(0, n)) - T
        
        for a in alpha_arr:
            # Doing bootstraping inside the k-fold sample data
            e = bootstrapping_lasso(30, x[list(S)], y[list(S)], a)
            
            if best_err is None or e < best_err:
                best_err = e
                best_a = a
                
        print("Fold #%d: Best error is %f" % (i, best_err))
        best_alphas.append(best_a)
        
    print(best_alphas)
    print("The best alpha is %f" % mode(best_alphas).mode)

if __name__ == "__main__":
    print('================== Ridge regression hyperparameter tuning ==============')
    hyper_tuning_ridge()
    print('================== Lasso regression hyperparameter tuning')
    hyper_tuning_lasso()