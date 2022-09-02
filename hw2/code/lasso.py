#!/usr/bin/env python3

import numpy as np
from setup_problem import load_problem
import matplotlib.pyplot as plt


def get_data():
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    X_train = featurize(x_train)
    X_val = featurize(x_val)

    return X_train, y_train, X_val, y_val


def soft(a, delta):
    if abs(a) - delta >= 0:
        return np.sign(a) * (abs(a) - delta)
    else:
        return 0


def lasso_objective(X, y, w, lambd):
    return np.sum((y - np.dot(X, w))**2) + lambd * sum([abs(ww) for ww in w])



def shooting_algorithm(X, y, lambd, num_iter=1000, tolerance=1e-6):
    n, D = X.shape
    # initialize w as 0 for all
    w = np.zeros(D)
    count = 0
    prev = lasso_objective(X, y, w, lambd)
    curr = 0

    while count < num_iter:
        for j in range(D):
            aj = 2 * np.dot(X[:,j].T, X[:,j])
            cj = 2 * (np.dot(X[:,j].T, y) - np.dot(np.dot(X[:,j].T, X), w) + w[j] * np.dot(X[:,j].T, X[:,j]))
            wj = soft(cj/aj, lambd/aj)
            w[j] = wj
        curr = lasso_objective(X, y, w, lambd)
        if prev - curr < tolerance:
            break
        prev = curr
        count += 1
    
    return w


def mse(X, y, w, b):
    n = X.shape[0]
    return 1./n * np.sum((np.dot(X, w) + b - y)**2)



def test_lasso():
    X_train, y_train, X_val, y_val = get_data()

    lambd_grid = np.unique(np.concatenate((10.**np.arange(-6,1,1), np.arange(1,3,.3) )))
    error_list = []
    for lambd in lambd_grid:
        w = shooting_algorithm(X_train, y_train, lambd)
        error = mse(X_val, y_val, w, b=0)
        error_list.append(error)
        print("lambd={}, mse={}".format(lambd, error))
    
    plt.plot(np.log10(lambd_grid), error_list)
    plt.xlabel('$\log_{10}\lambda$')
    plt.ylabel('validation error')
    plt.show()



def test_sklearn():
    from sklearn.linear_model import Lasso
    X_train, y_train, X_val, y_val = get_data()

    lambd_grid = np.unique(np.concatenate((10.**np.arange(-6,1,1), np.arange(1,3,.3) )))
    error_list = []
    for lambd in lambd_grid:
        model = Lasso(alpha=lambd)
        model.fit(X_train, y_train)
        error = mse(X_val, y_val, model.coef_, model.intercept_)
        error_list.append(error)
        print("lambd={}, mse={}".format(lambd, error))
    
    plt.plot(np.log10(lambd_grid), error_list)
    plt.xlabel('$\log_{10}\lambda$')
    plt.ylabel('validation error')
    plt.show()





def main():
    test_sklearn()



if __name__ == '__main__':
    main()
