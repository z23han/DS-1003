import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    num_instances, _ = X.shape
    summ = 0
    for r in range(num_instances):
        summ += np.logaddexp(0, -y[r] * np.dot(theta, X[r]))
    return 1./num_instances * summ + l2_param * np.dot(theta, theta)

    
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    _, num_features = X.shape

    theta0 = np.zeros(num_features)
    optimal_theta = minimize(objective_function, theta0, args=(X, y, l2_param)).x
    return optimal_theta


def load_data():
    X_train = pd.read_csv('X_train.txt', header=None).values
    X_val = pd.read_csv('X_val.txt', header=None).values
    y_train = pd.read_csv('y_train.txt', header=None).values
    y_val = pd.read_csv('y_val.txt', header=None).values

    return X_train, X_val, y_train, y_val


def preprocess(X_data):
    scaler = preprocessing.StandardScaler().fit(X_data)
    return scaler.transform(X_data)


def log_likelihood(X, y, theta):
    num_instances, _ = X.shape
    y = y.reshape(-1)
    summ = 0
    for r in range(num_instances):
        summ += (y[r] - 1) * np.dot(theta, X[r]) - np.logaddexp(0, -np.dot(theta, X[r]))
    return summ/num_instances


def process():
    X_train, X_val, y_train, y_val = load_data()

    X_train_bias = np.ones((X_train.shape[0], 1))
    X_val_bias = np.ones((X_val.shape[0], 1))
    X_train = np.hstack((X_train, X_train_bias))
    X_val = np.hstack((X_val, X_val_bias))

    X_train = preprocess(X_train)
    X_val = preprocess(X_val)

    l2_params = np.arange(-3, 5, 0.5)
    log_lk_res = []
    for l2_param in l2_params:
        print("processing l2_param={}".format(l2_param))
        theta = fit_logistic_reg(X_train, y_train, f_objective, l2_param=l2_param)
        log_lk = log_likelihood(X_val, y_val, theta)
        print("log_likelihood={}".format(log_lk))
        log_lk_res.append(log_lk)
    
    plt.plot(l2_params, np.array(log_lk_res))
    plt.xlabel("lambda")
    plt.ylabel("log_likelihood")
    plt.show()





def main():
    process()



if __name__ == '__main__':
    main()
